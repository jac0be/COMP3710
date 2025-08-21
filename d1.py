import torch
torch.cuda.empty_cache()
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def dragon_curve(iterations):
    # Start with two points forming a line segment
    points = torch.tensor([[0.0, 0.0],
                           [1.0, 0.0]], device=device)

    for _ in range(iterations):
        # Reverse and rotate the points (excluding last point)
        rev = points[:-1].flip(0)  # reverse excluding last point

        # Rotation matrix for 90 degrees CCW
        theta = torch.tensor([[0, -1],
                              [1,  0]], dtype=torch.float32, device=device)

        # Calculate pivot point (end of current curve)
        pivot = points[-1]

        # Shift reversed points to origin relative to pivot
        rev_shifted = rev - pivot

        # Rotate reversed points
        rev_rotated = torch.matmul(rev_shifted, theta.T)

        # Shift back to pivot
        rev_new = rev_rotated + pivot

        # Concatenate original points and rotated reversed points
        points = torch.cat([points, rev_new], dim=0)

    return points.cpu().numpy()

# Generate Dragon Curve points with 30 iterations
curve_points = dragon_curve(iterations=20)

# Plot the curve
plt.figure(figsize=(8,8))
plt.plot(curve_points[:,0], curve_points[:,1], color='blue')
plt.title('Dragon Curve Fractal (12 iterations)')
plt.axis('equal')
plt.axis('off')
plt.show()
