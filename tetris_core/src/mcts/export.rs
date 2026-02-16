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
    // Create the node (children will be filled in later)
    let export = TreeNodeExport {
        id,
        node_type: "decision".to_string(),
        visit_count: node.visit_count,
        value_sum: node.value_sum,
        mean_value: node.mean_value(),
        value_history: node.value_history.clone().unwrap_or_default(),
        nn_value: node.nn_value,
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

    let export = TreeNodeExport {
        id,
        node_type: "chance".to_string(),
        visit_count: node.visit_count,
        value_sum: node.value_sum,
        mean_value: node.mean_value(),
        value_history: node.value_history.clone().unwrap_or_default(),
        nn_value: node.nn_value,
        is_terminal: false,
        move_number: node.move_number,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::env::TetrisEnv;

    #[test]
    fn test_export_decision_node_sorts_children_and_sets_edges() {
        let env = TetrisEnv::with_seed(10, 20, 7);
        let mut root = DecisionNode::new(env.clone(), 0);
        root.visit_count = 4;
        root.value_sum = 10.0;

        let chance_high = ChanceNode::new(env.clone(), 3, 0, 0, vec![1, 2], 0.0, vec![0.5, 0.5]);
        let mut chance_low = ChanceNode::new(env.clone(), 1, 0, 0, vec![4, 5], 0.0, vec![0.5, 0.5]);

        // Insert out of order to verify deterministic sorting in export.
        let mut decision_piece_5 = DecisionNode::new(env.clone(), 1);
        decision_piece_5.visit_count = 2;
        decision_piece_5.value_sum = 5.0;
        let mut decision_piece_2 = DecisionNode::new(env.clone(), 1);
        decision_piece_2.visit_count = 1;
        decision_piece_2.value_sum = 1.0;
        chance_low
            .children
            .insert(5, MCTSNode::Decision(decision_piece_5));
        chance_low
            .children
            .insert(2, MCTSNode::Decision(decision_piece_2));

        root.children.insert(10, MCTSNode::Chance(chance_high));
        root.children.insert(1, MCTSNode::Chance(chance_low));

        let mut nodes = Vec::new();
        let root_id = export_decision_node(&root, None, None, &mut nodes);

        assert_eq!(root_id, 0);
        assert_eq!(nodes[0].node_type, "decision");
        assert_eq!(nodes[0].mean_value, 2.5);

        // Root children should be sorted by action index: 1 then 10.
        let root_child_edges: Vec<usize> = nodes[0]
            .children
            .iter()
            .map(|&id| nodes[id].edge_from_parent.unwrap())
            .collect();
        assert_eq!(root_child_edges, vec![1, 10]);

        // The chance node on action 1 should sort its piece children: 2 then 5.
        let chance_node_id = nodes[0].children[0];
        let chance_child_edges: Vec<usize> = nodes[chance_node_id]
            .children
            .iter()
            .map(|&id| nodes[id].edge_from_parent.unwrap())
            .collect();
        assert_eq!(chance_child_edges, vec![2, 5]);
    }
}
