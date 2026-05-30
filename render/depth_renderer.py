"""Multi-view depth map renderer for CMGR.

Renders V depth maps from a point cloud at 224x224 resolution
using Open3D's offscreen rendering pipeline.

Camera poses are generated using spherical coordinates with
uniformly distributed azimuth angles and fixed elevation.
"""

import torch
import torch.nn as nn
import numpy as np

try:
    import open3d as o3d
    import open3d.visualization.rendering as rendering
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    print("Warning: open3d not installed. DepthRenderer will use fallback mode.")


class MultiViewDepthRenderer(nn.Module):
    """Renders multi-view depth maps from 3D point clouds.

    Uses Open3D to render depth images from V viewpoints placed on a
    sphere around the object. Each view produces a 224x224 depth map.

    Camera setup:
    - V views with uniformly spaced azimuth angles [0, 360)
    - Fixed elevation angle (default 30 degrees)
    - Fixed distance from origin
    - Camera always looks at the origin
    """

    def __init__(self, num_views=12, resolution=224, points_radius=0.02,
                 elevation=30.0, distance=2.0):
        super().__init__()
        self.num_views = num_views
        self.resolution = resolution
        self.points_radius = points_radius
        self.elevation = elevation
        self.distance = distance

        # Precompute camera poses (azimuth angles)
        self.register_buffer(
            'azimuths',
            torch.linspace(0, 360.0, num_views + 1)[:-1]  # exclude 360 = 0
        )

    def render_single_view(self, points_np, azimuth, elevation=None, distance=None):
        """Render a single depth map from a point cloud.

        Args:
            points_np: [N, 3] numpy array of point coordinates (normalized).
            azimuth: Camera azimuth angle in degrees.
            elevation: Camera elevation angle in degrees (default: self.elevation).
            distance: Camera distance (default: self.distance).

        Returns:
            depth_map: [H, W] numpy array of depth values.
        """
        if elevation is None:
            elevation = self.elevation
        if distance is None:
            distance = self.distance

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_np.astype(np.float64))

        # Set point colors to white for rendering
        colors = np.ones_like(points_np) * 0.7
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # Create offscreen renderer
        renderer = o3d.visualization.rendering.OffscreenRenderer(
            self.resolution, self.resolution)

        # Setup material
        mat = rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        mat.point_size = self.points_radius * 1000

        renderer.scene.add_geometry("pointcloud", pcd, mat)

        # Compute camera position from spherical coordinates
        azim_rad = np.radians(azimuth)
        elev_rad = np.radians(elevation)

        cam_x = distance * np.cos(elev_rad) * np.sin(azim_rad)
        cam_y = distance * np.sin(elev_rad)
        cam_z = distance * np.cos(elev_rad) * np.cos(azim_rad)

        eye = np.array([cam_x, cam_y, cam_z])
        center = np.array([0.0, 0.0, 0.0])
        up = np.array([0.0, 1.0, 0.0])

        # Set camera
        renderer.setup_camera(
            60.0,  # field of view
            center,
            eye,
            up
        )

        # Render depth
        depth_image = renderer.render_to_depth_image()
        depth_np = np.asarray(depth_image)

        # Clean up
        del renderer

        return depth_np

    def normalize_point_cloud(self, points):
        """Normalize point cloud to fit within unit sphere.

        Args:
            points: [N, 3] numpy array.

        Returns:
            Normalized [N, 3] numpy array.
        """
        centroid = points.mean(axis=0)
        points = points - centroid
        max_dist = np.max(np.linalg.norm(points, axis=1))
        if max_dist > 0:
            points = points / max_dist
        return points

    def render_depth_maps(self, points):
        """Render multi-view depth maps for a single point cloud.

        Args:
            points: [N, 3] numpy array or tensor of point coordinates.

        Returns:
            depth_maps: [V, 1, H, W] torch tensor of depth maps.
        """
        if isinstance(points, torch.Tensor):
            points = points.cpu().numpy()

        # Normalize
        points = self.normalize_point_cloud(points)

        depth_maps = []
        for v in range(self.num_views):
            azim = float(self.azimuths[v].item()) if isinstance(self.azimuths[v], torch.Tensor) \
                else float(self.azimuths[v])
            depth = self.render_single_view(points, azim)

            # Replace inf/nan with 0 (background)
            depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)

            # Normalize depth to [0, 1] range
            depth_max = depth.max()
            if depth_max > 0:
                depth = depth / depth_max

            depth_maps.append(depth)

        depth_maps = np.stack(depth_maps, axis=0)  # [V, H, W]
        depth_maps = torch.from_numpy(depth_maps).float()
        depth_maps = depth_maps.unsqueeze(1)  # [V, 1, H, W]

        return depth_maps

    def forward(self, point_clouds):
        """Render depth maps for a batch of point clouds.

        Args:
            point_clouds: [B, N, 3] tensor of point cloud coordinates.

        Returns:
            depth_maps: [B, V, 1, H, W] tensor of depth maps.
        """
        batch_size = point_clouds.shape[0]
        all_depth_maps = []

        for b in range(batch_size):
            depth_maps = self.render_depth_maps(point_clouds[b])
            all_depth_maps.append(depth_maps)

        return torch.stack(all_depth_maps, dim=0)  # [B, V, 1, H, W]


class FallbackDepthRenderer(nn.Module):
    """Fallback depth renderer when Open3D is not available.

    Uses a simple z-buffer approach to render depth maps from point clouds
    without requiring Open3D's full rendering pipeline.
    """

    def __init__(self, num_views=12, resolution=224):
        super().__init__()
        self.num_views = num_views
        self.resolution = resolution

        # Precompute camera poses
        azimuths = torch.linspace(0, 360.0, num_views + 1)[:-1]
        self.register_buffer('azimuths', azimuths)

    def normalize_point_cloud(self, points):
        """Normalize point cloud to fit within unit sphere."""
        centroid = points.mean(dim=0, keepdim=True)
        points = points - centroid
        max_dist = torch.max(torch.norm(points, dim=1))
        if max_dist > 0:
            points = points / max_dist
        return points

    def project_points(self, points, azimuth, elevation=30.0, distance=2.0):
        """Project 3D points to 2D using perspective projection.

        Args:
            points: [N, 3] tensor.
            azimuth: Azimuth angle in degrees.
            elevation: Elevation angle in degrees.

        Returns:
            depth_map: [H, W] depth map tensor.
        """
        azim_rad = np.radians(azimuth)
        elev_rad = np.radians(elevation)

        # Rotation matrix (world-to-camera)
        cos_a, sin_a = np.cos(azim_rad), np.sin(azim_rad)
        cos_e, sin_e = np.cos(elev_rad), np.sin(elev_rad)

        # Camera position
        cam_x = distance * cos_e * sin_a
        cam_y = distance * sin_e
        cam_z = distance * cos_e * cos_a
        cam_pos = torch.tensor([cam_x, cam_y, cam_z], device=points.device, dtype=points.dtype)

        # View direction (camera -> origin)
        forward = -cam_pos / torch.norm(cam_pos)
        right = torch.cross(forward, torch.tensor([0.0, 1.0, 0.0],
                                                    device=points.device, dtype=points.dtype))
        right = right / torch.norm(right)
        up = torch.cross(right, forward)

        # Transform points to camera space
        points_cam = points - cam_pos.unsqueeze(0)  # [N, 3]
        depth = torch.sum(points_cam * forward.unsqueeze(0), dim=1)  # [N]

        # Project to image plane
        fov = 60.0
        focal = self.resolution / (2.0 * np.tan(np.radians(fov / 2.0)))

        x_img = torch.sum(points_cam * right.unsqueeze(0), dim=1) * focal / (depth + 1e-8)
        y_img = -torch.sum(points_cam * up.unsqueeze(0), dim=1) * focal / (depth + 1e-8)

        # Convert to pixel coordinates
        x_pixel = (x_img + self.resolution / 2).long()
        y_pixel = (y_img + self.resolution / 2).long()

        # Create depth map
        depth_map = torch.zeros(self.resolution, self.resolution, device=points.device)
        depth_valid = (depth > 0) & \
                      (x_pixel >= 0) & (x_pixel < self.resolution) & \
                      (y_pixel >= 0) & (y_pixel < self.resolution)

        for i in range(points.shape[0]):
            if depth_valid[i]:
                px, py = x_pixel[i], y_pixel[i]
                if depth_map[py, px] == 0 or depth[i] < depth_map[py, px]:
                    depth_map[py, px] = depth[i]

        # Normalize
        depth_max = depth_map.max()
        if depth_max > 0:
            depth_map = depth_map / depth_max

        return depth_map

    def forward(self, point_clouds):
        """Render depth maps for a batch of point clouds.

        Args:
            point_clouds: [B, N, 3] tensor.

        Returns:
            depth_maps: [B, V, 1, H, W] tensor.
        """
        batch_size = point_clouds.shape[0]
        all_depth_maps = []

        for b in range(batch_size):
            points = self.normalize_point_cloud(point_clouds[b])
            depth_maps = []
            for v in range(self.num_views):
                azim = float(self.azimuths[v].item())
                depth = self.project_points(points, azim)
                depth_maps.append(depth)
            depth_maps = torch.stack(depth_maps, dim=0)  # [V, H, W]
            depth_maps = depth_maps.unsqueeze(1)  # [V, 1, H, W]
            all_depth_maps.append(depth_maps)

        return torch.stack(all_depth_maps, dim=0)  # [B, V, 1, H, W]


def create_depth_renderer(num_views=12, resolution=224, use_open3d=True):
    """Factory function to create the appropriate depth renderer.

    Args:
        num_views: Number of views to render.
        resolution: Image resolution (H=W).
        use_open3d: If True, try to use Open3D renderer first.

    Returns:
        DepthRenderer instance.
    """
    if use_open3d and HAS_OPEN3D:
        try:
            return MultiViewDepthRenderer(num_views=num_views, resolution=resolution)
        except Exception as e:
            print(f"Open3D renderer failed ({e}), using fallback renderer.")
            return FallbackDepthRenderer(num_views=num_views, resolution=resolution)
    else:
        return FallbackDepthRenderer(num_views=num_views, resolution=resolution)
