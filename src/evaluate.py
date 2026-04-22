import sys
sys.path.append(".")
from dotenv import load_dotenv
from src.graph import build_graph

load_dotenv()

TEST_QUESTIONS = [
    "What is knowledge distillation and how does it compress neural networks?",
    "What are the main techniques for neural network pruning?",
    "How does quantization reduce the size of deep learning models?",
    "What is the difference between structured and unstructured pruning?",
    "What is the lottery ticket hypothesis?",
]

def run_evaluation():
    agent = build_graph()
    results = []

    print("Evaluando agente...\n")
    for question in TEST_QUESTIONS:
        result = agent.invoke({
            "question": question,
            "documents": [],
            "generation": "",
            "is_relevant": False
        })
        results.append({
            "question": question,
            "is_relevant": result["is_relevant"],
            "answer_length": len(result["generation"]),
            "answer_preview": result["generation"][:200]
        })
        status = "✓" if result["is_relevant"] else "✗"
        print(f"{status} {question[:60]}...")

    relevant = sum(1 for r in results if r["is_relevant"])
    print(f"\n=== RESULTADOS ===")
    print(f"Relevancia: {relevant}/{len(results)} ({relevant/len(results)*100:.0f}%)")

    import csv
    with open("data/evaluation_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print("Resultados guardados en data/evaluation_results.csv")

if __name__ == "__main__":
    run_evaluation()