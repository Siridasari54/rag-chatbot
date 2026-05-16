# evaluation/eval_dataset.py — NEW FILE

"""
Ground truth QA dataset for evaluating DeepLens retrieval quality.
Add your own questions and answers based on your actual uploaded PDFs.
"""

EVAL_DATASET = [
    {
        "question": "What is machine learning?",
        "ground_truth": "Machine learning is a subset of artificial intelligence that enables systems to learn from data and improve from experience without being explicitly programmed.",
    },
    {
        "question": "What is deep learning?",
        "ground_truth": "Deep learning is a subset of machine learning that uses neural networks with multiple layers to learn representations of data.",
    },
    {
        "question": "What is backpropagation?",
        "ground_truth": "Backpropagation is a training algorithm used in neural networks that calculates gradients of the loss function with respect to weights by propagating errors backward through the network.",
    },
    {
        "question": "What is a neural network?",
        "ground_truth": "A neural network is a computational model inspired by the human brain, consisting of interconnected nodes organized in layers that process information.",
    },
    {
        "question": "What is overfitting?",
        "ground_truth": "Overfitting occurs when a model learns the training data too well including noise and performs poorly on new unseen data.",
    },
    {
        "question": "What is gradient descent?",
        "ground_truth": "Gradient descent is an optimization algorithm that minimizes the loss function by iteratively moving in the direction of steepest descent.",
    },
    {
        "question": "What is NLP?",
        "ground_truth": "Natural Language Processing is a branch of AI that deals with the interaction between computers and human language.",
    },
    {
        "question": "What is a CNN?",
        "ground_truth": "A Convolutional Neural Network is a deep learning architecture designed for processing structured grid data like images.",
    },
    {
        "question": "What is transfer learning?",
        "ground_truth": "Transfer learning is a technique where a model trained on one task is reused as the starting point for a model on a different task.",
    },
    {
        "question": "What is an RNN?",
        "ground_truth": "A Recurrent Neural Network is a neural network designed for sequential data where connections between nodes form a directed graph along a temporal sequence.",
    },
]


def run_ground_truth_eval(chain, llm) -> list:
    """
    Run all eval questions through the chain and score against ground truth.
    Returns list of result dicts.
    """
    results = []
    for item in EVAL_DATASET:
        try:
            result = chain.invoke({"question": item["question"]})
            answer = result.get("answer", "")

            # simple word overlap score against ground truth
            gt_words  = set(item["ground_truth"].lower().split())
            ans_words = set(answer.lower().split())
            overlap   = len(gt_words & ans_words) / max(len(gt_words), 1)

            results.append({
                "question":     item["question"],
                "ground_truth": item["ground_truth"],
                "answer":       answer,
                "overlap_score": round(overlap, 2),
                "retrieved":    "isn't covered" not in answer.lower(),
            })
        except Exception as e:
            results.append({
                "question":     item["question"],
                "ground_truth": item["ground_truth"],
                "answer":       f"ERROR: {e}",
                "overlap_score": 0,
                "retrieved":    False,
            })
    return results