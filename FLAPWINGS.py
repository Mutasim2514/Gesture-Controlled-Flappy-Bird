import cv2
import mediapipe as mp
import numpy as np
import time
import math
import pygame
import random
import threading
from pygame.locals import *

# GAME VARIABLES
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 600
SPEED = 20
GRAVITY = 2.5
GAME_SPEED = 15

GROUND_WIDTH = 2 * SCREEN_WIDTH
GROUND_HEIGHT = 100

PIPE_WIDTH = 80
PIPE_HEIGHT = 600
PIPE_GAP = 250

# Audio files
wing = 'assets/audio/wing.wav'
hit = 'assets/audio/hit.wav'


pygame.mixer.init()


class FlapDetector:
    def __init__(self):

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Pose landmarks for arms
        self.LEFT_SHOULDER = 11
        self.RIGHT_SHOULDER = 12
        self.LEFT_ELBOW = 13
        self.RIGHT_ELBOW = 14
        self.LEFT_WRIST = 15
        self.RIGHT_WRIST = 16

        # Fixed flap detection variables
        self.baseline_arm_height = None
        self.calibration_frames = 0
        self.calibration_needed = 30
        self.flap_threshold = 0.12  # Increased threshold for more intentional flaps
        self.is_flapping = False
        self.flap_cooldown = 0
        self.flap_cooldown_max = 15  # Increased cooldown to prevent double-triggers

        # Simple position tracking
        self.arm_positions = []
        self.max_history = 5
        self.last_arm_position = None

        # State tracking
        self.arms_raised = False
        self.waiting_for_down = False

        # Shared flap state
        self.flap_detected = False
        self.running = True
        self.camera_ready = False

    def get_arm_height(self, landmarks):
        """Calculate average height of both arms (elbows)"""
        if not landmarks:
            return None

        # Use elbows for more reliable detection
        left_elbow_y = landmarks[self.LEFT_ELBOW].y
        right_elbow_y = landmarks[self.RIGHT_ELBOW].y

        # Average elbow height (lower y = higher position)
        arm_height = (left_elbow_y + right_elbow_y) / 2
        return arm_height

    def detect_flap(self, current_arm_height):
        """Simplified flap detection - look for clear up-down motion"""
        if current_arm_height is None:
            return False

        # Calibration phase
        if self.calibration_frames < self.calibration_needed:
            if self.baseline_arm_height is None:
                self.baseline_arm_height = current_arm_height
            else:
                # Smooth baseline update
                self.baseline_arm_height = 0.9 * self.baseline_arm_height + 0.1 * current_arm_height
            self.calibration_frames += 1
            return False

        # Handle cooldown
        if self.flap_cooldown > 0:
            self.flap_cooldown -= 1

        # Calculate how much arms are raised
        height_diff = self.baseline_arm_height - current_arm_height
        arms_significantly_raised = height_diff > self.flap_threshold

        # State machine for flap detection
        if not self.arms_raised and arms_significantly_raised:
            # Arms just went up
            self.arms_raised = True
            self.waiting_for_down = False

        elif self.arms_raised and not arms_significantly_raised:
            # Arms came back down - this is a complete flap!
            if self.flap_cooldown == 0:
                self.arms_raised = False
                self.flap_cooldown = self.flap_cooldown_max
                return True
            else:
                self.arms_raised = False

        return False

    def process_frame(self, frame):
        """Process frame for flap detection"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)

        flap_detected = False

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            current_arm_height = self.get_arm_height(landmarks)
            flap_detected = self.detect_flap(current_arm_height)

            # Draw pose landmarks for arms
            h, w = frame.shape[:2]
            arm_points = [self.LEFT_SHOULDER, self.RIGHT_SHOULDER,
                          self.LEFT_ELBOW, self.RIGHT_ELBOW,
                          self.LEFT_WRIST, self.RIGHT_WRIST]

            # Draw arm skeleton
            connections = [
                (self.LEFT_SHOULDER, self.LEFT_ELBOW),
                (self.LEFT_ELBOW, self.LEFT_WRIST),
                (self.RIGHT_SHOULDER, self.RIGHT_ELBOW),
                (self.RIGHT_ELBOW, self.RIGHT_WRIST)
            ]

            # Draw connections
            for start_idx, end_idx in connections:
                if (landmarks[start_idx].visibility > 0.5 and
                        landmarks[end_idx].visibility > 0.5):
                    start_x = int(landmarks[start_idx].x * w)
                    start_y = int(landmarks[start_idx].y * h)
                    end_x = int(landmarks[end_idx].x * w)
                    end_y = int(landmarks[end_idx].y * h)
                    cv2.line(frame, (start_x, start_y), (end_x, end_y), (0, 255, 255), 3)

            # Draw joint points
            for idx in arm_points:
                if landmarks[idx].visibility > 0.5:
                    x = int(landmarks[idx].x * w)
                    y = int(landmarks[idx].y * h)
                    color = (0, 255, 0) if idx in [self.LEFT_WRIST, self.RIGHT_WRIST] else (255, 0, 0)
                    cv2.circle(frame, (x, y), 8, color, -1)

        # Draw UI
        self.draw_ui(frame, flap_detected)

        # Update shared state
        if flap_detected:
            self.flap_detected = True

        return frame

    def draw_ui(self, frame, flap_detected):
        """Draw UI overlay"""
        h, w = frame.shape[:2]

        # Status text
        if self.calibration_frames < self.calibration_needed:
            status = f"Calibrating... {self.calibration_frames}/{self.calibration_needed}"
            color = (255, 255, 0)
        else:
            status = "Ready! Raise arms UP then DOWN to flap!"
            color = (0, 255, 0)

        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Arm state indicator
        if self.calibration_frames >= self.calibration_needed:
            if self.arms_raised:
                state_text = "ARMS UP - now bring them DOWN!"
                state_color = (255, 255, 0)
            else:
                state_text = "Arms down - raise UP to prepare flap"
                state_color = (255, 255, 255)

            cv2.putText(frame, state_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, state_color, 2)

        # Cooldown indicator
        if self.flap_cooldown > 0:
            cd_text = f"Cooldown: {self.flap_cooldown}"
            cv2.putText(frame, cd_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Instructions
        cv2.putText(frame, "Motion: UP then DOWN = FLAP", (10, h - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, "Press 'q' to quit", (10, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Flap indicator
        if flap_detected:
            cv2.putText(frame, "FLAP DETECTED!", (w // 2 - 100, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 4)

    def start_camera(self):
        """Start camera processing"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Signal that camera is ready
        self.camera_ready = True

        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            processed_frame = self.process_frame(frame)

            cv2.imshow("Wing Flap Detector", processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                break

        cap.release()
        cv2.destroyAllWindows()

    def get_flap_state(self):
        """Get and reset flap state"""
        if self.flap_detected:
            self.flap_detected = False
            return True
        return False

    def stop(self):
        """Stop the detector"""
        self.running = False
        if hasattr(self, 'pose'):
            self.pose.close()


class Bird(pygame.sprite.Sprite):
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)

        # Original bird images
        self.images = [pygame.image.load('assets/sprites/bluebird-upflap.png').convert_alpha(),
                       pygame.image.load('assets/sprites/bluebird-midflap.png').convert_alpha(),
                       pygame.image.load('assets/sprites/bluebird-downflap.png').convert_alpha()]

        self.speed = SPEED
        self.current_image = 0
        self.image = pygame.image.load('assets/sprites/bluebird-upflap.png').convert_alpha()
        self.mask = pygame.mask.from_surface(self.image)

        self.rect = self.image.get_rect()
        self.rect[0] = SCREEN_WIDTH / 6
        self.rect[1] = SCREEN_HEIGHT / 2

        # Animation timing - FIXED
        self.animation_timer = 0

    def update(self):
        # FIXED: Proper bird animation
        self.animation_timer += 1
        if self.animation_timer >= 5:  # Change every 5 frames
            self.current_image = (self.current_image + 1) % 3
            self.image = self.images[self.current_image]
            self.animation_timer = 0

        self.speed += GRAVITY
        # UPDATE HEIGHT
        self.rect[1] += self.speed

    def bump(self):
        self.speed = -SPEED
        # Reset animation on flap
        self.current_image = 0
        self.image = self.images[0]
        self.animation_timer = 0

    def begin(self):
        # FIXED: Slower animation for begin screen
        self.animation_timer += 1
        if self.animation_timer >= 10:  # Slower animation
            self.current_image = (self.current_image + 1) % 3
            self.image = self.images[self.current_image]
            self.animation_timer = 0


class Pipe(pygame.sprite.Sprite):
    def __init__(self, inverted, xpos, ysize):
        pygame.sprite.Sprite.__init__(self)

        self.image = pygame.image.load('assets/sprites/pipe-green.png').convert_alpha()
        self.image = pygame.transform.scale(self.image, (PIPE_WIDTH, PIPE_HEIGHT))

        self.rect = self.image.get_rect()
        self.rect[0] = xpos

        if inverted:
            self.image = pygame.transform.flip(self.image, False, True)
            self.rect[1] = -(self.rect[3] - ysize)
        else:
            self.rect[1] = SCREEN_HEIGHT - ysize

        self.mask = pygame.mask.from_surface(self.image)

    def update(self):
        self.rect[0] -= GAME_SPEED


class Ground(pygame.sprite.Sprite):
    def __init__(self, xpos):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load('assets/sprites/base.png').convert_alpha()
        self.image = pygame.transform.scale(self.image, (GROUND_WIDTH, GROUND_HEIGHT))

        self.mask = pygame.mask.from_surface(self.image)

        self.rect = self.image.get_rect()
        self.rect[0] = xpos
        self.rect[1] = SCREEN_HEIGHT - GROUND_HEIGHT

    def update(self):
        self.rect[0] -= GAME_SPEED


def is_off_screen(sprite):
    return sprite.rect[0] < -(sprite.rect[2])


def get_random_pipes(xpos):
    size = random.randint(100, 300)
    pipe = Pipe(False, xpos, size)
    pipe_inverted = Pipe(True, xpos, SCREEN_HEIGHT - size - PIPE_GAP)
    return pipe, pipe_inverted


def reset_game():
    """Reset all game elements for a new game"""
    # Create new bird
    bird_group = pygame.sprite.Group()
    bird = Bird()
    bird_group.add(bird)

    # Create new ground
    ground_group = pygame.sprite.Group()
    for i in range(2):
        ground = Ground(GROUND_WIDTH * i)
        ground_group.add(ground)

    # Create new pipes
    pipe_group = pygame.sprite.Group()
    for i in range(2):
        pipes = get_random_pipes(SCREEN_WIDTH * i + 800)
        pipe_group.add(pipes[0])
        pipe_group.add(pipes[1])

    return bird_group, ground_group, pipe_group, bird


def main():
    # Initialize flap detector
    flap_detector = FlapDetector()

    # Start camera thread immediately
    camera_thread = threading.Thread(target=flap_detector.start_camera)
    camera_thread.daemon = True
    camera_thread.start()

    # Initialize pygame
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption('Wing Flap Controlled Flappy Bird')

    # Original game assets
    BACKGROUND = pygame.image.load('assets/sprites/background-day.png')
    BACKGROUND = pygame.transform.scale(BACKGROUND, (SCREEN_WIDTH, SCREEN_HEIGHT))
    BEGIN_IMAGE = pygame.image.load('assets/sprites/message.png').convert_alpha()

    # Create font for messages
    font = pygame.font.Font(None, 36)
    small_font = pygame.font.Font(None, 24)

    clock = pygame.time.Clock()

    print("🐦 Wing Flap Controlled Flappy Bird Started!")
    print("🎥 Camera starting simultaneously...")
    print("Flap motion: Raise arms UP, then bring them DOWN = FLAP!")
    print("Press 'q' in camera window or close game window to quit")

    # Wait a moment for camera to initialize
    while not flap_detector.camera_ready and flap_detector.running:
        pygame.time.wait(10)

    # Main game loop with restart functionality
    while flap_detector.running:
        # Initialize/Reset game elements
        bird_group, ground_group, pipe_group, bird = reset_game()
        begin = True
        score = 0
        game_over = False

        # Begin screen
        while begin and flap_detector.running:
            clock.tick(15)

            for event in pygame.event.get():
                if event.type == QUIT:
                    flap_detector.stop()
                    pygame.quit()
                    return
                if event.type == KEYDOWN:
                    if event.key == K_SPACE or event.key == K_UP:
                        bird.bump()
                        pygame.mixer.music.load(wing)
                        pygame.mixer.music.play()
                        begin = False

            # Check for flap
            if flap_detector.get_flap_state():
                bird.bump()
                pygame.mixer.music.load(wing)
                pygame.mixer.music.play()
                begin = False

            screen.blit(BACKGROUND, (0, 0))
            screen.blit(BEGIN_IMAGE, (120, 150))

            # Add instruction text
            instruction_text = small_font.render("Raise arms UP then DOWN to flap!", True, (255, 255, 255))
            screen.blit(instruction_text, (SCREEN_WIDTH // 2 - instruction_text.get_width() // 2, 450))

            if is_off_screen(ground_group.sprites()[0]):
                ground_group.remove(ground_group.sprites()[0])
                new_ground = Ground(GROUND_WIDTH - 20)
                ground_group.add(new_ground)

            bird.begin()
            ground_group.update()

            bird_group.draw(screen)
            ground_group.draw(screen)

            pygame.display.update()

        # Main game loop
        while flap_detector.running and not game_over:
            clock.tick(15)

            for event in pygame.event.get():
                if event.type == QUIT:
                    flap_detector.stop()
                    pygame.quit()
                    return
                if event.type == KEYDOWN:
                    if event.key == K_SPACE or event.key == K_UP:
                        bird.bump()
                        pygame.mixer.music.load(wing)
                        pygame.mixer.music.play()

            # Check for flap
            if flap_detector.get_flap_state():
                bird.bump()
                pygame.mixer.music.load(wing)
                pygame.mixer.music.play()

            screen.blit(BACKGROUND, (0, 0))

            # Update ground
            if is_off_screen(ground_group.sprites()[0]):
                ground_group.remove(ground_group.sprites()[0])
                new_ground = Ground(GROUND_WIDTH - 20)
                ground_group.add(new_ground)

            # Update pipes and score
            if is_off_screen(pipe_group.sprites()[0]):
                pipe_group.remove(pipe_group.sprites()[0])
                pipe_group.remove(pipe_group.sprites()[0])
                pipes = get_random_pipes(SCREEN_WIDTH * 2)
                pipe_group.add(pipes[0])
                pipe_group.add(pipes[1])
                score += 1

            # Update sprites
            bird_group.update()
            ground_group.update()
            pipe_group.update()

            # Draw sprites
            bird_group.draw(screen)
            pipe_group.draw(screen)
            ground_group.draw(screen)

            # Draw score
            score_text = font.render(f"Score: {score}", True, (255, 255, 255))
            screen.blit(score_text, (10, 10))

            pygame.display.update()

            # Check collisions
            if (pygame.sprite.groupcollide(bird_group, ground_group, False, False, pygame.sprite.collide_mask) or
                    pygame.sprite.groupcollide(bird_group, pipe_group, False, False, pygame.sprite.collide_mask)):
                pygame.mixer.music.load(hit)
                pygame.mixer.music.play()
                game_over = True

        # Game Over screen
        if flap_detector.running:
            game_over_time = time.time()
            restart_wait = False

            while flap_detector.running and (time.time() - game_over_time < 3):
                clock.tick(15)

                for event in pygame.event.get():
                    if event.type == QUIT:
                        flap_detector.stop()
                        pygame.quit()
                        return

                # Draw game over screen
                screen.blit(BACKGROUND, (0, 0))
                bird_group.draw(screen)
                pipe_group.draw(screen)
                ground_group.draw(screen)

                # Game over overlay
                overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
                overlay.set_alpha(128)
                overlay.fill((0, 0, 0))
                screen.blit(overlay, (0, 0))

                # Game over text
                game_over_text = font.render("GAME OVER", True, (255, 0, 0))
                final_score_text = font.render(f"Final Score: {score}", True, (255, 255, 255))
                restart_text = small_font.render("Restarting in 3 seconds...", True, (255, 255, 255))

                # Center the text
                screen.blit(game_over_text,
                            (SCREEN_WIDTH // 2 - game_over_text.get_width() // 2, SCREEN_HEIGHT // 2 - 60))
                screen.blit(final_score_text,
                            (SCREEN_WIDTH // 2 - final_score_text.get_width() // 2, SCREEN_HEIGHT // 2 - 20))
                screen.blit(restart_text, (SCREEN_WIDTH // 2 - restart_text.get_width() // 2, SCREEN_HEIGHT // 2 + 20))

                pygame.display.update()

            # Continue to next iteration (restart)
            continue

    # Cleanup
    flap_detector.stop()
    pygame.quit()


if __name__ == "__main__":
    main()