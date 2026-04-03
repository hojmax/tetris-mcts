use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use crate::game::constants::{BOARD_HEIGHT, BOARD_WIDTH};
use crate::replay::ReplayMove;

use super::config::MCTSConfig;
use super::results::{GameTreePlayback, MCTSTreeExport, TreeNodeExport};

pub(crate) const TREE_PLAYBACK_FORMAT_VERSION: u32 = 1;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct SavedPlaybackConfig {
    pub num_simulations: u32,
    pub c_puct: f32,
    pub temperature: f32,
    pub dirichlet_alpha: f32,
    pub dirichlet_epsilon: f32,
    pub visit_sampling_epsilon: f32,
    pub max_placements: u32,
    pub reuse_tree: bool,
    pub use_parent_value_for_unvisited_q: bool,
    pub nn_value_weight: f32,
    pub death_penalty: f32,
    pub overhang_penalty_weight: f32,
    pub mcts_seed: Option<u64>,
}

impl From<&MCTSConfig> for SavedPlaybackConfig {
    fn from(config: &MCTSConfig) -> Self {
        Self {
            num_simulations: config.num_simulations,
            c_puct: config.c_puct,
            temperature: config.temperature,
            dirichlet_alpha: config.dirichlet_alpha,
            dirichlet_epsilon: config.dirichlet_epsilon,
            visit_sampling_epsilon: config.visit_sampling_epsilon,
            max_placements: config.max_placements,
            reuse_tree: config.reuse_tree,
            use_parent_value_for_unvisited_q: config.use_parent_value_for_unvisited_q,
            nn_value_weight: config.nn_value_weight,
            death_penalty: config.death_penalty,
            overhang_penalty_weight: config.overhang_penalty_weight,
            mcts_seed: config.seed,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct SavedTreeNode {
    pub id: usize,
    pub node_type: String,
    pub visit_count: u32,
    pub value_sum: f32,
    pub mean_value: f32,
    pub value_history: Vec<f32>,
    pub nn_value: f32,
    pub unvisited_child_value_estimate: f32,
    pub is_terminal: bool,
    pub move_number: u32,
    pub attack: u32,
    pub parent_id: Option<usize>,
    pub edge_from_parent: Option<usize>,
    pub children: Vec<usize>,
    pub valid_actions: Vec<usize>,
    pub action_priors: Vec<f32>,
}

impl From<&TreeNodeExport> for SavedTreeNode {
    fn from(node: &TreeNodeExport) -> Self {
        Self {
            id: node.id,
            node_type: node.node_type.clone(),
            visit_count: node.visit_count,
            value_sum: node.value_sum,
            mean_value: node.mean_value,
            value_history: node.value_history.clone(),
            nn_value: node.nn_value,
            unvisited_child_value_estimate: node.unvisited_child_value_estimate,
            is_terminal: node.is_terminal,
            move_number: node.move_number,
            attack: node.attack,
            parent_id: node.parent_id,
            edge_from_parent: node.edge_from_parent,
            children: node.children.clone(),
            valid_actions: node.valid_actions.clone(),
            action_priors: node.action_priors.clone(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct SavedTreeExport {
    pub nodes: Vec<SavedTreeNode>,
    pub root_id: usize,
    pub num_simulations: u32,
    pub selected_action: usize,
    pub policy: Vec<f32>,
    pub q_min: f32,
    pub q_max: f32,
}

impl From<&MCTSTreeExport> for SavedTreeExport {
    fn from(tree: &MCTSTreeExport) -> Self {
        Self {
            nodes: tree.nodes.iter().map(SavedTreeNode::from).collect(),
            root_id: tree.root_id,
            num_simulations: tree.num_simulations,
            selected_action: tree.selected_action,
            policy: tree.policy.clone(),
            q_min: tree.q_min,
            q_max: tree.q_max,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct SavedGameTreeStep {
    pub frame_index: u32,
    pub placement_count: u32,
    pub selected_action: usize,
    pub selected_chance_outcome: usize,
    pub attack: u32,
    pub tree: SavedTreeExport,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct SavedGameTreeMetadata {
    pub format_version: u32,
    pub source: String,
    pub initial_seed: u64,
    pub board_width: usize,
    pub board_height: usize,
    pub add_noise: bool,
    pub model_path: String,
    pub candidate_step: u64,
    pub promoted: bool,
    pub candidate_avg_attack: f32,
    pub evaluation_seconds: f32,
    pub search_config: SavedPlaybackConfig,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct SavedGameTreePlayback {
    pub metadata: SavedGameTreeMetadata,
    pub replay_moves: Vec<ReplayMove>,
    pub steps: Vec<SavedGameTreeStep>,
    pub total_attack: u32,
    pub num_moves: u32,
    pub num_frames: u32,
    pub tree_reuse_hits: u32,
    pub tree_reuse_misses: u32,
}

impl SavedGameTreePlayback {
    pub(crate) fn from_playback(
        playback: &GameTreePlayback,
        initial_seed: u64,
        config: &MCTSConfig,
        add_noise: bool,
        model_path: &Path,
        candidate_step: u64,
        promoted: bool,
        candidate_avg_attack: f32,
        evaluation_seconds: f32,
    ) -> Self {
        Self {
            metadata: SavedGameTreeMetadata {
                format_version: TREE_PLAYBACK_FORMAT_VERSION,
                source: "candidate_eval_worst_game".to_string(),
                initial_seed,
                board_width: BOARD_WIDTH,
                board_height: BOARD_HEIGHT,
                add_noise,
                model_path: model_path.display().to_string(),
                candidate_step,
                promoted,
                candidate_avg_attack,
                evaluation_seconds,
                search_config: SavedPlaybackConfig::from(config),
            },
            replay_moves: playback.replay_moves.clone(),
            steps: playback
                .steps
                .iter()
                .map(|step| SavedGameTreeStep {
                    frame_index: step.frame_index,
                    placement_count: step.placement_count,
                    selected_action: step.selected_action,
                    selected_chance_outcome: step.selected_chance_outcome,
                    attack: step.attack,
                    tree: SavedTreeExport::from(&step.tree),
                })
                .collect(),
            total_attack: playback.total_attack,
            num_moves: playback.num_moves,
            num_frames: playback.num_frames,
            tree_reuse_hits: playback.tree_reuse_hits,
            tree_reuse_misses: playback.tree_reuse_misses,
        }
    }
}

pub(crate) fn persist_saved_game_tree_playback(
    playback: &SavedGameTreePlayback,
    path: &Path,
) -> Result<(), String> {
    let parent = path.parent().ok_or_else(|| {
        format!(
            "Tree playback output path has no parent directory: {}",
            path.display()
        )
    })?;
    fs::create_dir_all(parent).map_err(|error| {
        format!(
            "Failed to create tree playback directory {}: {}",
            parent.display(),
            error
        )
    })?;

    let temp_path = temporary_output_path(path);
    let file = File::create(&temp_path).map_err(|error| {
        format!(
            "Failed to create temporary tree playback file {}: {}",
            temp_path.display(),
            error
        )
    })?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer(&mut writer, playback).map_err(|error| {
        format!(
            "Failed to serialize tree playback {}: {}",
            temp_path.display(),
            error
        )
    })?;
    writer.flush().map_err(|error| {
        format!(
            "Failed to flush tree playback {}: {}",
            temp_path.display(),
            error
        )
    })?;

    fs::rename(&temp_path, path).map_err(|error| {
        format!(
            "Failed to move temporary tree playback {} into place at {}: {}",
            temp_path.display(),
            path.display(),
            error
        )
    })?;
    Ok(())
}

fn temporary_output_path(path: &Path) -> PathBuf {
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("tree_playback.json");
    path.with_file_name(format!("{file_name}.tmp"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::constants::{BOARD_HEIGHT, BOARD_WIDTH};
    use crate::game::env::TetrisEnv;
    use crate::runtime::test_utils;
    use crate::search::{MCTSAgent, MCTSConfig};

    #[test]
    fn test_saved_game_tree_playback_serializes_compact_nodes() {
        let mut config = MCTSConfig::default();
        config.num_simulations = 25;
        config.max_placements = 4;
        config.reuse_tree = true;

        let agent = MCTSAgent::new(config.clone());
        let env = TetrisEnv::with_seed(BOARD_WIDTH, BOARD_HEIGHT, 17);
        let playback = agent
            .play_game_with_trees(&env, 4, false)
            .expect("playback should succeed");

        let saved = SavedGameTreePlayback::from_playback(
            &playback,
            17,
            &config,
            false,
            Path::new("candidate.onnx"),
            123,
            false,
            0.0,
            1.25,
        );

        assert_eq!(saved.metadata.initial_seed, 17);
        assert_eq!(saved.metadata.search_config.num_simulations, 25);
        assert_eq!(saved.steps.len(), playback.steps.len());
        assert_eq!(saved.replay_moves.len(), playback.replay_moves.len());
        assert!(saved
            .steps
            .iter()
            .flat_map(|step| step.tree.nodes.iter())
            .all(|node| !node.node_type.is_empty()));
    }

    #[test]
    fn test_persist_saved_game_tree_playback_writes_json_file() {
        let output_path =
            test_utils::unique_temp_path("tree_playback", "saved").with_extension("json");
        let playback = SavedGameTreePlayback {
            metadata: SavedGameTreeMetadata {
                format_version: TREE_PLAYBACK_FORMAT_VERSION,
                source: "test".to_string(),
                initial_seed: 9,
                board_width: BOARD_WIDTH,
                board_height: BOARD_HEIGHT,
                add_noise: false,
                model_path: "model.onnx".to_string(),
                candidate_step: 7,
                promoted: true,
                candidate_avg_attack: 12.0,
                evaluation_seconds: 0.5,
                search_config: SavedPlaybackConfig {
                    num_simulations: 10,
                    c_puct: 1.5,
                    temperature: 0.0,
                    dirichlet_alpha: 0.03,
                    dirichlet_epsilon: 0.25,
                    visit_sampling_epsilon: 0.0,
                    max_placements: 4,
                    reuse_tree: true,
                    use_parent_value_for_unvisited_q: false,
                    nn_value_weight: 0.01,
                    death_penalty: 0.0,
                    overhang_penalty_weight: 0.0,
                    mcts_seed: Some(0),
                },
            },
            replay_moves: Vec::new(),
            steps: Vec::new(),
            total_attack: 0,
            num_moves: 0,
            num_frames: 0,
            tree_reuse_hits: 0,
            tree_reuse_misses: 0,
        };

        persist_saved_game_tree_playback(&playback, &output_path).expect("persist should succeed");
        let serialized = fs::read_to_string(&output_path).expect("json file should exist");
        let restored: SavedGameTreePlayback =
            serde_json::from_str(&serialized).expect("json should deserialize");
        assert_eq!(restored.metadata.initial_seed, 9);
        assert_eq!(restored.metadata.candidate_step, 7);
        let _ = fs::remove_file(output_path);
    }
}
