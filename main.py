from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from agents import analyze_paper, find_gaps, connect_papers, synthesize_gaps, score_connections
from typing import TypedDict
from pathlib import Path
from datetime import datetime

load_dotenv()

class ResearchState(TypedDict):
    pdf_paths: list
    analyses: list
    gaps: list
    connections: str
    synthesis: str
    edge_scores: list

graph = StateGraph(ResearchState)


graph.add_node("analyze_paper", analyze_paper)
graph.add_node("find_gaps", find_gaps)
graph.add_node("connect_papers", connect_papers)
graph.add_node("synthesize_gaps", synthesize_gaps)
graph.add_node("score_connections", score_connections)

graph.add_edge(START, "analyze_paper")
graph.add_edge("analyze_paper", "find_gaps")
graph.add_edge("find_gaps", "connect_papers")
graph.add_edge("connect_papers", "synthesize_gaps")
graph.add_edge("synthesize_gaps", "score_connections")
graph.add_edge("score_connections", END)

app = graph.compile()

def run_pipeline(pdf_paths: list) -> dict:
    result = app.invoke({
        "pdf_paths": pdf_paths,
        "analyses": [],
        "gaps": [],
        "connections": "",
        "synthesis": "",
        "edge_scores": []
    })
    return {
        "analyses": result["analyses"],
        "gaps": result["gaps"],
        "connections": result["connections"],
        "synthesis": result["synthesis"],
        "edge_scores": result["edge_scores"]
    }


if __name__ == "__main__":
    result = run_pipeline([
        "C:\\Users\\Lenovo\\phase3-multi-agent\\paper1.pdf",
        "C:\\Users\\Lenovo\\phase3-multi-agent\\paper2.pdf"
    ])
    print("Edge scores:", result["edge_scores"])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("reports")
    output_dir.mkdir(exist_ok=True)

    report_path = output_dir / f"research_report_{timestamp}.md"

    report = f"""# Research Gap Analysis Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

## Paper Analyses

"""

    for i, analysis in enumerate(result["analyses"]):
        report += f"### Paper {i+1}\n{analysis}\n\n---\n\n"

    report += "## Research Gaps\n\n"
    for gap in result["gaps"]:
        report += f"{gap}\n\n---\n\n"

    report += f"## Cross-Paper Connections\n\n{result['connections']}\n\n---\n\n"
    report += f"## Synthesized Research Gaps\n\n{result['synthesis']}\n"

    report_path.write_text(report, encoding="utf-8")

    print(f"Report saved to: {report_path}")