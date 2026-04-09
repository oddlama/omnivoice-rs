use omnivoice_infer::{phase1_workspace_marker, workspace_phase_marker};

#[test]
fn workspace_phase_marker_tracks_phase10_baseline() {
    assert_eq!(workspace_phase_marker(), "omnivoice-phase10");
}

#[test]
fn legacy_phase1_marker_alias_points_to_current_phase() {
    assert_eq!(phase1_workspace_marker(), workspace_phase_marker());
}
