//! MCTS Tree Export
//!
//! Functions for exporting the MCTS tree structure for visualization.

use super::nodes::{ChanceNode, DecisionNode, MCTSNode};
use super::results::TreeNodeExport;

/// Recursively export a decision node and its subtree
pub(super) fn export_decision_node(
    node: &DecisionNode,
    parent_id: Option<usize>,
    edge_from_parent: Option<usize>,
    nodes: &mut Vec<TreeNodeExport>,
) -> usize {
    let id = nodes.len();
    let mean_value = if node.visit_count > 0 {
        node.value_sum / node.visit_count as f32
    } else {
        0.0
    };

    // Create the node (children will be filled in later)
    let export = TreeNodeExport {
        id,
        node_type: "decision".to_string(),
        visit_count: node.visit_count,
        value_sum: node.value_sum,
        mean_value,
        nn_value: node.nn_value,
        prior: node.prior,
        is_terminal: node.is_terminal,
        move_number: node.move_number,
        attack: 0,
        state: node.state.clone(),
        parent_id,
        edge_from_parent,
        children: Vec::new(),
        valid_actions: node.valid_actions.clone(),
        action_priors: node.action_priors.clone(),
    };

    nodes.push(export);

    // Export children (sorted by action index for deterministic order)
    let mut child_keys: Vec<usize> = node.children.keys().copied().collect();
    child_keys.sort();

    let mut child_ids = Vec::new();
    for action_idx in child_keys {
        if let Some(MCTSNode::Chance(chance_node)) = node.children.get(&action_idx) {
            let child_id = export_chance_node(chance_node, Some(id), Some(action_idx), nodes);
            child_ids.push(child_id);
        }
    }

    // Update our children list
    nodes[id].children = child_ids;

    id
}

/// Recursively export a chance node and its subtree
pub(super) fn export_chance_node(
    node: &ChanceNode,
    parent_id: Option<usize>,
    edge_from_parent: Option<usize>,
    nodes: &mut Vec<TreeNodeExport>,
) -> usize {
    let id = nodes.len();
    let mean_value = if node.visit_count > 0 {
        node.value_sum / node.visit_count as f32
    } else {
        0.0
    };

    let export = TreeNodeExport {
        id,
        node_type: "chance".to_string(),
        visit_count: node.visit_count,
        value_sum: node.value_sum,
        mean_value,
        nn_value: node.nn_value,
        prior: 0.0,
        is_terminal: false,
        move_number: 0,
        attack: node.attack,
        state: node.state.clone(),
        parent_id,
        edge_from_parent,
        children: Vec::new(),
        valid_actions: Vec::new(),
        action_priors: Vec::new(),
    };

    nodes.push(export);

    // Export children (sorted by piece type for deterministic order)
    let mut child_keys: Vec<usize> = node.children.keys().copied().collect();
    child_keys.sort();

    let mut child_ids = Vec::new();
    for piece_type in child_keys {
        if let Some(MCTSNode::Decision(decision_node)) = node.children.get(&piece_type) {
            let child_id = export_decision_node(decision_node, Some(id), Some(piece_type), nodes);
            child_ids.push(child_id);
        }
    }

    nodes[id].children = child_ids;

    id
}
