import math

import torch


class RandomRotationAugmentation:
    """Apply random rotation augmentation to a batch of 3D graphs.
    Each graph in the batch is rotated independently using a randomly sampled unit quaternion.
    """

    def __init__(self):
        pass

    def rotate_molecule_randomly(
        self, positions: torch.Tensor, batch_idx: torch.Tensor
    ) -> torch.Tensor:
        """Rotate and translate the positions of nodes in a batch of PyG graphs w.r.t. their centers.

        Args:
            positions (torch.Tensor): A tensor of shape (num_nodes, 3) containing the 3D positions of all nodes.
            batch_idx (torch.Tensor): A tensor of shape (num_nodes,) indicating the graph index for each node.

        Returns:
            torch.Tensor: A tensor of shape (num_nodes, 3) containing the rotated and translated positions.
        """
        # Step 0: Draw random quaternions for each graph
        batch_size = batch_idx.max().item() + 1
        device = positions.device
        rot_operator = self._draw_random_quaternions(batch_size, device)

        # Step 1: Convert quaternions to rotation matrices
        rot_operator = self._quaternions_to_rotation_matrices(
            rot_operator.clone()
        )  # Shape: (batch_size, 3, 3)

        # Step 2: Compute the center of each graph
        graph_centers = self._compute_graph_centers(
            positions, batch_idx
        )  # Shape: (batch_size, 3)

        # Step 3: Center the positions (subtract the graph center)
        centered_positions = (
            positions - graph_centers[batch_idx]
        )  # Shape: (num_nodes, 3)

        # Step 4: Apply rotation
        node_rotation_matrices = rot_operator[batch_idx]  # Shape: (num_nodes, 3, 3)
        rotated_positions = torch.bmm(
            node_rotation_matrices, centered_positions.unsqueeze(-1)
        ).squeeze(
            -1
        )  # Shape: (num_nodes, 3)

        # Step 5: Recenter the positions (add the graph center back)
        recentered_positions = (
            rotated_positions + graph_centers[batch_idx]
        )  # Shape: (num_nodes, 3)

        return recentered_positions

    def rotate_ligand_and_pocket_randomly(
        self,
        ligand_pos: torch.Tensor,
        ligand_batch: torch.Tensor,
        pocket_pos: torch.Tensor,
        pocket_batch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the same rigid transform as :meth:`rotate_graphs_randomly` to ligand and pocket.

        Rotation axes and angles are sampled per graph from ``ligand_batch``. Centers are
        the mean ligand position per graph so the pocket follows the ligand frame.
        """
        batch_size = int(ligand_batch.max().item()) + 1
        device = ligand_pos.device
        quaternions = self._draw_random_quaternions(batch_size, device)
        rot_operator = self._quaternions_to_rotation_matrices(quaternions.clone())
        graph_centers = self._compute_graph_centers(ligand_pos, ligand_batch)

        def _apply(pos: torch.Tensor, batch_idx: torch.Tensor) -> torch.Tensor:
            centered = pos - graph_centers[batch_idx]
            node_rotation_matrices = rot_operator[batch_idx]
            rotated = torch.bmm(node_rotation_matrices, centered.unsqueeze(-1)).squeeze(
                -1
            )
            return rotated + graph_centers[batch_idx]

        return _apply(ligand_pos, ligand_batch), _apply(pocket_pos, pocket_batch)

    def _quaternions_to_rotation_matrices(self, quaternions: torch.Tensor):
        """Convert a batch of quaternions to a batch of 3x3 rotation matrices.

        Args:
            quaternions (torch.Tensor): A tensor of shape (batch_size, 4) representing a batch of quaternions [q_w, q_x, q_y, q_z].
                                        The quaternions are assumed to be normalized.

        Returns:
            torch.Tensor: A tensor of shape (batch_size, 3, 3) containing the rotation matrices.
        """
        # Ensure the quaternions are normalized
        quaternions = quaternions / quaternions.norm(dim=1, keepdim=True)

        # Extract components of the quaternions
        q_w, q_x, q_y, q_z = (
            quaternions[:, 0],
            quaternions[:, 1],
            quaternions[:, 2],
            quaternions[:, 3],
        )

        # Compute the rotation matrices
        rotation_matrices = torch.stack(
            [
                torch.stack(
                    [
                        1 - 2 * (q_y**2 + q_z**2),
                        2 * (q_x * q_y - q_z * q_w),
                        2 * (q_x * q_z + q_y * q_w),
                    ],
                    dim=-1,
                ),
                torch.stack(
                    [
                        2 * (q_x * q_y + q_z * q_w),
                        1 - 2 * (q_x**2 + q_z**2),
                        2 * (q_y * q_z - q_x * q_w),
                    ],
                    dim=-1,
                ),
                torch.stack(
                    [
                        2 * (q_x * q_z - q_y * q_w),
                        2 * (q_y * q_z + q_x * q_w),
                        1 - 2 * (q_x**2 + q_y**2),
                    ],
                    dim=-1,
                ),
            ],
            dim=1,
        )  # Shape: (batch_size, 3, 3)

        return rotation_matrices

    def _compute_graph_centers(self, positions: torch.Tensor, batch_idx: torch.Tensor):
        """Compute the center (mean position) of each graph in the batch.

        Args:
            positions (torch.Tensor): A tensor of shape (num_nodes, 3) containing the 3D positions of all nodes.
            batch_idx (torch.Tensor): A tensor of shape (num_nodes,) indicating the graph index for each node.

        Returns:
            torch.Tensor: A tensor of shape (batch_size, 3) containing the center of each graph.
        """
        # Sum the positions for each graph
        graph_sums = torch.zeros(
            batch_idx.max() + 1, 3, device=positions.device, dtype=positions.dtype
        )
        graph_counts = torch.zeros(
            batch_idx.max() + 1, device=positions.device, dtype=positions.dtype
        )

        graph_sums.index_add_(0, batch_idx, positions)  # Sum positions for each graph
        graph_counts.index_add_(
            0, batch_idx, torch.ones_like(batch_idx, dtype=positions.dtype)
        )  # Count nodes per graph

        # Compute the mean (center) for each graph
        graph_centers = graph_sums / graph_counts.unsqueeze(1)  # Shape: (batch_size, 3)

        return graph_centers

    def _generate_uniform_quaternions(self, batch_size: int, device: torch.device):
        """Generate uniformly sampled unit quaternions.

        Args:
            batch_size (int): Number of quaternions to generate.
            device (torch.device): Device for computations (e.g., 'cpu' or 'cuda').

        Returns:
            torch.Tensor: Tensor of shape (batch_size, 4) containing unit quaternions.
        """
        N = batch_size

        # Step 1: Generate random rotation axes
        random_axes = torch.randn(N, 3, device=device)
        # Add a small epsilon to the norm to avoid division by zero
        epsilon = 1e-8
        random_axes = random_axes / (
            random_axes.norm(dim=1, keepdim=True) + epsilon
        )  # Normalize to unit vectors

        # Step 2: Generate random rotation angles
        random_angles = (torch.rand(N, device=device) * (math.pi - 0.1)) + (
            0.05 * math.pi
        )

        # Step 3: Construct random quaternions
        half_angles = random_angles / 2
        qw = torch.cos(half_angles)
        qxyz = random_axes * torch.sin(half_angles).unsqueeze(1)
        random_quaternions = torch.cat([qw.unsqueeze(1), qxyz], dim=1)  # Shape (N, 4)

        return random_quaternions

    def _draw_random_quaternions(self, batch_size: int, device: torch.device):
        """Draw random rotation quaternions for each graph.

        Args:
            batch_size (int): Number of quaternions to generate.
            device (torch.device): Device for computations.

        Returns:
            torch.Tensor: Tensor of shape (batch_size, 4) containing unit quaternions.
        """
        # Generate random rotation quaternions for each graph
        rot = self._generate_uniform_quaternions(batch_size, device)
        rot = rot / rot.norm(dim=1, keepdim=True)
        mask = rot[:, 0] < 0  # Check where w < 0
        rot[mask] = -rot[mask]  # Flip the sign of the entire quaternion

        return rot  # Shape: (batch_size, 4)
