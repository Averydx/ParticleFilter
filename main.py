import pygame
import numpy as np;
import scipy as scipy

pygame.init();
screen_size = (400, 400);
pygame.display.set_caption("Particle Filter");
framerate = 60;
mouse_pos = (0,0);
mouse_pos_prev = (0,0);
clock = pygame.time.Clock();
background = pygame.Surface((800,800));
landmarks_surface = pygame.Surface((800,800))
screen = pygame.display.set_mode((800,800));

pygame.draw.circle(landmarks_surface,center = (150,150),color = (255,0,0), radius = 10);
pygame.draw.circle(landmarks_surface,center = (650,150),color = (255,0,0), radius = 10);
pygame.draw.circle(landmarks_surface,center = (650,650),color = (255,0,0), radius = 10);
pygame.draw.circle(landmarks_surface,center = (150,650),color = (255,0,0), radius = 10);

measurements = np.array([0,0,0,0]);

landmarks = np.array([(150,150),(650,150),(650,650),(150,650)]);

def create_samples():
    samples = [];
    for i in range(400):
        samples.append((np.random.uniform(0,800),np.random.uniform(0,800)));
    samples = np.array(samples);
    return samples;

def create_weights():
    weights = [];
    for i in range(400):
        weights.append(1);
    weights = np.array(weights);
    return weights;

def particle_filter():
    move_vec = tuple(map(lambda i, j: i - j, mouse_pos, mouse_pos_prev));
    move_vec = np.array(move_vec);

    for i,landmark in enumerate(landmarks):
        distance = np.sqrt(np.power(mouse_pos[0] - landmark[0],2) + np.power(mouse_pos[1] - landmark[1],2));
        measurements[i] = distance;

    return move_vec;

def predict(samples,move_vec):
    for sample in samples:
        sample[0] += move_vec[0] + np.random.normal(0,5);
        sample[1] += move_vec[1] + np.random.normal(0,5);

def update(samples,weights):
    weights = np.ones(400);
    for i, landmark in enumerate(landmarks):
        distance = np.power((samples[:, 0] - landmark[0]) ** 2 + (samples[:, 1] - landmark[1]) ** 2, 0.5);
        weights *= scipy.stats.norm(distance, 50).pdf(measurements[i]);
    weights += 1.e-300  # avoid round-off to zero
    weights /= sum(weights)
    return weights;

def resample(samples,weights):
    indexes = np.zeros(400);
    for i in range(400):
        indexes[i] = i;
    new_sample_indexes = np.random.choice(a=indexes, size=400, replace=True, p=weights);
    sample_copy = np.copy(samples);
    for i in range(len(samples)):
        samples[i] = sample_copy[int(new_sample_indexes[i])];
    return samples;

weights = create_weights();

samples = create_samples();

while True:
    background.fill((0,0,0));
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit()

    mouse_pos_prev = mouse_pos;
    mouse_pos = pygame.mouse.get_pos();

    move_vec = particle_filter();
    predict(samples,move_vec);
    weights = update(samples,weights);
    samples = resample(samples,weights);
    screen.blit(pygame.transform.scale(background,(800,800)),(0,0));
    screen.blit(pygame.transform.scale(landmarks_surface, (800, 800)), (0, 0));
    pygame.draw.circle(screen, center=mouse_pos, color=(0, 255, 0), radius=5);
    for sample in samples:
        pygame.draw.circle(screen,center = sample, color = (255,255,255), radius = 1);
    pygame.display.update();
    clock.tick(framerate);







