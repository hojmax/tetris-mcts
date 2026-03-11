use super::*;
use crate::game::constants::{BOARD_HEIGHT, BOARD_WIDTH, NUM_PIECE_TYPES, ROW_FILL_FEATURE_ROWS};
use crate::runtime::test_utils;
use crate::search::NUM_ACTIONS;
use std::fs;

fn make_example(move_number: u32) -> TrainingExample {
    let mut policy = vec![0.0; NUM_ACTIONS];
    policy[0] = 1.0;
    let mut action_mask = vec![false; NUM_ACTIONS];
    action_mask[0] = true;

    TrainingExample {
        board: vec![0; BOARD_HEIGHT * BOARD_WIDTH],
        current_piece: 0,
        hold_piece: 7,
        hold_available: true,
        next_queue: vec![0, 1, 2, 3, 4],
        move_number,
        placement_count: move_number as f32 / 100.0,
        combo: 0.0,
        back_to_back: false,
        next_hidden_piece_probs: vec![1.0 / NUM_PIECE_TYPES as f32; NUM_PIECE_TYPES],
        column_heights: vec![0.0; BOARD_WIDTH],
        max_column_height: 0.0,
        row_fill_counts: vec![0.0; ROW_FILL_FEATURE_ROWS],
        total_blocks: 0.0,
        bumpiness: 0.0,
        holes: 0.0,
        policy,
        value: move_number as f32,
        action_mask,
        overhang_fields: crate::search::normalize_overhang_fields(move_number),
        game_number: 0,
        game_total_attack: 0,
    }
}

fn unique_temp_path(name: &str) -> PathBuf {
    test_utils::unique_temp_path("tetris_generator", name)
}

#[test]
fn test_shared_buffer_fifo_eviction() {
    let buffer = SharedBuffer::new(3);
    buffer.add_examples(vec![make_example(1), make_example(2)]);
    buffer.add_examples(vec![make_example(3), make_example(4)]);

    let (_, _, kept) = buffer.logical_window_snapshot().unwrap();
    assert_eq!(kept.len(), 3);
    let move_numbers: Vec<u32> = kept.iter().map(|e| e.move_number).collect();
    assert_eq!(move_numbers, vec![2, 3, 4]);
}

#[test]
fn test_shared_buffer_logical_indices_follow_fifo_window() {
    let buffer = SharedBuffer::new(3);
    assert!(buffer.logical_window_snapshot().is_none());

    buffer.add_examples(vec![make_example(1), make_example(2)]);
    let (window_start, window_end, snapshot) = buffer
        .logical_window_snapshot()
        .expect("snapshot should exist after adding examples");
    assert_eq!(window_start, 0);
    assert_eq!(window_end, 2);
    let move_numbers: Vec<u32> = snapshot.iter().map(|e| e.move_number).collect();
    assert_eq!(move_numbers, vec![1, 2]);

    buffer.add_examples(vec![make_example(3), make_example(4)]);
    let (window_start, window_end, snapshot) = buffer
        .logical_window_snapshot()
        .expect("snapshot should exist after eviction");
    assert_eq!(window_start, 1);
    assert_eq!(window_end, 4);
    let move_numbers: Vec<u32> = snapshot.iter().map(|e| e.move_number).collect();
    assert_eq!(move_numbers, vec![2, 3, 4]);

    let (window_start, window_end, slice_start, slice) = buffer
        .logical_delta_slice(2, 10)
        .expect("delta should exist");
    assert_eq!(window_start, 1);
    assert_eq!(window_end, 4);
    assert_eq!(slice_start, 2);
    let move_numbers: Vec<u32> = slice.iter().map(|e| e.move_number).collect();
    assert_eq!(move_numbers, vec![3, 4]);

    let (window_start, window_end, slice_start, slice) = buffer
        .logical_delta_slice(0, 2)
        .expect("delta should clamp to window start");
    assert_eq!(window_start, 1);
    assert_eq!(window_end, 4);
    assert_eq!(slice_start, 1);
    let move_numbers: Vec<u32> = slice.iter().map(|e| e.move_number).collect();
    assert_eq!(move_numbers, vec![2, 3]);
}

#[test]
fn test_shared_stats_accumulates_and_tracks_max_combo() {
    let stats = SharedStats::new();
    let game_a = GameStats {
        singles: 1,
        doubles: 0,
        triples: 0,
        tetrises: 0,
        tspin_minis: 0,
        tspin_singles: 0,
        tspin_doubles: 0,
        tspin_triples: 0,
        perfect_clears: 0,
        back_to_backs: 1,
        max_combo: 2,
        total_lines: 1,
        holds: 3,
    };
    let game_b = GameStats {
        singles: 0,
        doubles: 1,
        triples: 0,
        tetrises: 1,
        tspin_minis: 0,
        tspin_singles: 1,
        tspin_doubles: 0,
        tspin_triples: 0,
        perfect_clears: 1,
        back_to_backs: 0,
        max_combo: 5,
        total_lines: 6,
        holds: 4,
    };

    stats.add(&game_a, 0);
    stats.add(&game_b, 7);
    let d = stats.to_dict();

    assert_eq!(d["games_with_attack"], 1);
    assert_eq!(d["games_with_lines"], 2);
    assert_eq!(d["singles"], 1);
    assert_eq!(d["doubles"], 1);
    assert_eq!(d["tetrises"], 1);
    assert_eq!(d["tspin_singles"], 1);
    assert_eq!(d["perfect_clears"], 1);
    assert_eq!(d["back_to_backs"], 1);
    assert_eq!(d["max_combo"], 5);
    assert_eq!(d["total_lines"], 7);
    assert_eq!(d["total_attack"], 7);
    assert_eq!(d["holds"], 7);
}

#[test]
fn test_build_candidate_eval_config_forces_deterministic_gate_settings() {
    let mut config = MCTSConfig::default();
    config.seed = None;
    config.visit_sampling_epsilon = 0.37;
    config.num_simulations = 321;
    config.temperature = 0.9;
    config.dirichlet_alpha = 0.02;
    config.dirichlet_epsilon = 0.4;

    let eval_config = GameGenerator::build_candidate_eval_config(&config);
    assert_eq!(eval_config.seed, Some(0));
    assert_eq!(eval_config.visit_sampling_epsilon, 0.0);
    assert_eq!(eval_config.num_simulations, 321);
    assert_eq!(eval_config.temperature, 0.9);
    assert_eq!(eval_config.dirichlet_alpha, 0.02);
    assert_eq!(eval_config.dirichlet_epsilon, 0.4);
}

#[test]
fn test_effective_search_penalties_disable_penalties_at_cap() {
    assert_eq!(
        GameGenerator::effective_search_penalties(0.99, 1.0, 5.0, 3.0),
        (5.0, 3.0)
    );
    assert_eq!(
        GameGenerator::effective_search_penalties(1.0, 1.0, 5.0, 3.0),
        (0.0, 0.0)
    );
    assert_eq!(
        GameGenerator::effective_search_penalties(1.2, 1.0, 5.0, 3.0),
        (0.0, 0.0)
    );
}

#[test]
fn test_incumbent_eval_rebaseline_only_when_penalties_change_for_network_incumbent() {
    assert!(!GameGenerator::needs_incumbent_eval_rebaseline(
        false,
        (5.0, 3.0),
        (0.0, 0.0),
    ));
    assert!(!GameGenerator::needs_incumbent_eval_rebaseline(
        true,
        (5.0, 3.0),
        (5.0, 3.0),
    ));
    assert!(GameGenerator::needs_incumbent_eval_rebaseline(
        true,
        (5.0, 3.0),
        (0.0, 0.0),
    ));
}

#[test]
fn test_build_rollout_config_keeps_sampling_settings() {
    let mut config = MCTSConfig::default();
    config.num_simulations = 123;
    config.visit_sampling_epsilon = 0.42;
    config.temperature = 1.5;
    config.dirichlet_alpha = 0.02;
    config.dirichlet_epsilon = 0.3;
    config.nn_value_weight = 0.123;
    config.death_penalty = 5.0;
    config.overhang_penalty_weight = 3.0;
    config.q_scale = Some(7.5);

    let network_config =
        GameGenerator::build_rollout_config(&config, true, 999, true, 0.123, 5.0, 3.0);
    assert_eq!(network_config.num_simulations, 123);
    assert_eq!(network_config.visit_sampling_epsilon, 0.42);
    assert_eq!(network_config.temperature, 1.5);
    assert_eq!(network_config.dirichlet_alpha, 0.02);
    assert_eq!(network_config.dirichlet_epsilon, 0.3);
    assert_eq!(network_config.nn_value_weight, 0.123);
    assert_eq!(network_config.death_penalty, 5.0);
    assert_eq!(network_config.overhang_penalty_weight, 3.0);
    assert_eq!(network_config.q_scale, Some(7.5));

    let bootstrap_config =
        GameGenerator::build_rollout_config(&config, false, 999, true, 0.456, 5.0, 3.0);
    assert_eq!(bootstrap_config.num_simulations, 999);
    assert_eq!(bootstrap_config.visit_sampling_epsilon, 0.42);
    assert_eq!(bootstrap_config.temperature, 1.5);
    assert_eq!(bootstrap_config.dirichlet_alpha, 0.02);
    assert_eq!(bootstrap_config.dirichlet_epsilon, 0.3);
    assert_eq!(bootstrap_config.nn_value_weight, 0.456);
    assert_eq!(bootstrap_config.death_penalty, 5.0);
    assert_eq!(bootstrap_config.overhang_penalty_weight, 3.0);
    assert_eq!(bootstrap_config.q_scale, None);

    // Verify penalties are overridden when passed as 0
    let zeroed_config =
        GameGenerator::build_rollout_config(&config, true, 999, true, 1.0, 0.0, 0.0);
    assert_eq!(zeroed_config.death_penalty, 0.0);
    assert_eq!(zeroed_config.overhang_penalty_weight, 0.0);

    // When disabled, bootstrap keeps the configured q_scale.
    let bootstrap_no_force =
        GameGenerator::build_rollout_config(&config, false, 999, false, 0.456, 5.0, 3.0);
    assert_eq!(bootstrap_no_force.q_scale, Some(7.5));
}

#[test]
fn test_model_artifact_paths_cover_all_split_outputs() {
    let onnx_path = unique_temp_path("candidate").with_extension("onnx");
    let artifacts = GameGenerator::model_artifact_paths(&onnx_path);
    assert_eq!(artifacts[0], onnx_path);
    assert_eq!(artifacts[1], onnx_path.with_extension("onnx.data"));
    assert_eq!(artifacts[2], onnx_path.with_extension("conv.onnx"));
    assert_eq!(artifacts[3], onnx_path.with_extension("conv.onnx.data"));
    assert_eq!(artifacts[4], onnx_path.with_extension("heads.onnx"));
    assert_eq!(artifacts[5], onnx_path.with_extension("heads.onnx.data"));
    assert_eq!(artifacts[6], onnx_path.with_extension("fc.bin"));
}

#[test]
fn test_remove_model_artifacts_if_safe_preserves_bootstrap_and_incumbent() {
    let model_path = unique_temp_path("cleanup").with_extension("onnx");
    let artifacts = GameGenerator::model_artifact_paths(&model_path);
    for artifact in &artifacts {
        fs::write(artifact, b"model").expect("temp file write should succeed");
    }

    GameGenerator::remove_model_artifacts_if_safe(
        &model_path,
        &model_path,
        &PathBuf::from("different.onnx"),
        None,
    );
    for artifact in &artifacts {
        assert!(
            artifact.exists(),
            "bootstrap artifacts should not be removed"
        );
    }

    GameGenerator::remove_model_artifacts_if_safe(
        &model_path,
        &PathBuf::from("bootstrap.onnx"),
        &model_path,
        None,
    );
    for artifact in &artifacts {
        assert!(
            artifact.exists(),
            "incumbent artifacts should not be removed"
        );
    }

    GameGenerator::remove_model_artifacts_if_safe(
        &model_path,
        &PathBuf::from("bootstrap.onnx"),
        &PathBuf::from("incumbent.onnx"),
        None,
    );
    for artifact in &artifacts {
        assert!(!artifact.exists(), "candidate artifacts should be removed");
    }
}

#[test]
fn test_cleanup_queued_candidate_artifacts_removes_pending_and_evaluating() {
    let bootstrap_path = unique_temp_path("bootstrap").with_extension("onnx");
    let training_data_path = unique_temp_path("training_data");
    let generator = GameGenerator::new(
        bootstrap_path.to_string_lossy().to_string(),
        training_data_path.to_string_lossy().to_string(),
        None,
        100,
        true,
        16,
        1.0,
        1,
        0,
        Some(vec![0]),
        true,
        10,
        true,
        0.0,
        1.0,
    )
    .expect("generator should construct");

    let pending_path = unique_temp_path("pending").with_extension("onnx");
    for artifact in GameGenerator::model_artifact_paths(&pending_path) {
        fs::write(&artifact, b"pending").expect("pending artifact write should succeed");
    }
    let evaluating_path = unique_temp_path("evaluating").with_extension("onnx");
    for artifact in GameGenerator::model_artifact_paths(&evaluating_path) {
        fs::write(&artifact, b"evaluating").expect("evaluating artifact write should succeed");
    }

    {
        let mut pending = generator.pending_candidate.write().unwrap();
        *pending = Some(CandidateModelRequest {
            model_path: pending_path.clone(),
            model_step: 1,
            nn_value_weight: 0.1,
        });
    }
    {
        let mut evaluating = generator.evaluating_candidate.write().unwrap();
        *evaluating = Some(CandidateModelRequest {
            model_path: evaluating_path.clone(),
            model_step: 2,
            nn_value_weight: 0.2,
        });
    }

    generator.cleanup_queued_candidate_artifacts();

    for artifact in GameGenerator::model_artifact_paths(&pending_path) {
        assert!(!artifact.exists(), "pending artifacts should be removed");
    }
    for artifact in GameGenerator::model_artifact_paths(&evaluating_path) {
        assert!(!artifact.exists(), "evaluating artifacts should be removed");
    }
    assert!(generator.pending_candidate.read().unwrap().is_none());
    assert!(generator.evaluating_candidate.read().unwrap().is_none());
}
