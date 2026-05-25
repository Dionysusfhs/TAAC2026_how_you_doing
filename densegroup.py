def _build_user_dense_group_layout(
    user_dense_feature_specs: Optional[List[Tuple[int, int, int]]]
) -> List[Tuple[str, List[Tuple[int, int]], int]]:
    """Build user dense groups from schema entries.

    Special dense embeddings use fid=61 and fid=87. All other user dense
    fids are concatenated into the normal statistics branch.
    """
    if not user_dense_feature_specs:
        return []

    special_fid_to_group = {
        61: 'emb61',
        87: 'emb87',
    }
    normal_segments: List[Tuple[int, int]] = []
    special_segments = {group_name: [] for group_name in special_fid_to_group.values()}

    for fid, offset, length in user_dense_feature_specs:
        group_name = special_fid_to_group.get(int(fid))
        segment = (offset, length)
        if group_name is None:
            normal_segments.append(segment)
        else:
            special_segments[group_name].append(segment)

    layout: List[Tuple[str, List[Tuple[int, int]], int]] = []
    normal_dim = sum(length for _, length in normal_segments)
    if normal_dim > 0:
        layout.append(('normal', normal_segments, normal_dim))
    for group_name in ('emb61', 'emb87'):
        group_segments = special_segments[group_name]
        group_dim = sum(length for _, length in group_segments)
        if group_dim > 0:
            layout.append((group_name, group_segments, group_dim))
    return layout

@staticmethod
def _slice_dense_segments(
    dense_feats: torch.Tensor,
    segments: List[Tuple[int, int]],
) -> torch.Tensor:
    """Slice one or more dense segments and concatenate them in schema order."""
    if len(segments) == 1:
        offset, length = segments[0]
        return dense_feats[:, offset:offset + length]
    return torch.cat(
        [dense_feats[:, offset:offset + length] for offset, length in segments],
        dim=1,
    )

def _build_user_dense_token(self, user_dense_feats: torch.Tensor) -> torch.Tensor:
    """Build the user dense NS token with optional grouped projections."""
    if self.use_user_dense_groups:
        projected_groups = []
        for (_, group_segments), proj in zip(self._user_dense_group_layout, self.user_dense_group_projs):
            group_feats = self._slice_dense_segments(user_dense_feats, group_segments)
            projected_groups.append(proj(group_feats))
        fused_dense = projected_groups[0]
        for group_tok in projected_groups[1:]:
            fused_dense = fused_dense + group_tok
        return F.silu(fused_dense).unsqueeze(1)
    return F.silu(self.user_dense_proj(user_dense_feats)).unsqueeze(1)