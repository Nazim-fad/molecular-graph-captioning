"""
Logic for constructing the prompt inputs for the LLM.
"""

def graph_to_summary_text(graph):
    """
    Converts graph features (PyG Data object) into a text summary.
    OGB-style features in graph.x.
    """
    x = graph.x

    # Basic stats
    num_atoms = x.size(0)
    num_bonds = graph.edge_index.size(1) // 2

    # Feature extraction for summary
    atomic_nums = x[:, 0]
    formal_charges = x[:, 3]
    is_aromatic = x[:, 7].bool()
    is_in_ring = x[:, 8].bool()

    total_charge = int(formal_charges.sum().item())
    has_aromatic = bool(is_aromatic.any().item())
    has_ring = bool(is_in_ring.any().item())

    # Specific atom checks
    num_heteroatoms = int((atomic_nums != 6).sum().item())
    has_phosphorus = bool((atomic_nums == 15).any().item())
    has_nitrogen = bool((atomic_nums == 7).any().item())

    summary = (
        f"Target molecule graph summary:\n"
        f"- Number of atoms: {num_atoms}\n"
        f"- Number of bonds: {num_bonds}\n"
        f"- Contains aromatic atoms: {'yes' if has_aromatic else 'no'}\n"
        f"- Contains rings: {'yes' if has_ring else 'no'}\n"
        f"- Total formal charge: {total_charge}\n"
        f"- Number of heteroatoms: {num_heteroatoms}\n"
        f"- Contains phosphorus: {'yes' if has_phosphorus else 'no'}\n"
        f"- Contains nitrogen: {'yes' if has_nitrogen else 'no'}\n"
    )

    return summary


def build_prompt(graph, retrieved_texts):
    """
    Constructs the final prompt string.
    
    Args:
        graph: PyG Data object (target molecule)
        retrieved_texts: List[str] of descriptions from similar molecules
    """
    graph_summary = graph_to_summary_text(graph)

    prompt = (
        "You are a chemistry expert.\n\n"
        f"{graph_summary}\n"
        "Below are descriptions of molecules that are structurally and "
        "functionally similar to the target molecule:\n\n"
    )

    for i, txt in enumerate(retrieved_texts, 1):
        prompt += f"[{i}] {txt}\n"

    prompt += (
        "\nBased on the graph summary and the retrieved examples above, "
        "write a concise and accurate description of the target molecule.\n\n"
        "Description:"
    )
    return prompt