from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from tools import build_paper_index, query_paper
import time

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

def analyze_paper(state: dict) -> dict:
    analyses = []
    for pdf_path in state["pdf_paths"]:
        try:
            index = build_paper_index(pdf_path)
            problem = query_paper(index, "core problem research question motivation")
            novelty = query_paper(index, "novelty contribution proposed method solution")
            response = llm.invoke([HumanMessage(content=(
                "You are a research paper analyst.\n"
                "Based on these relevant excerpts from a research paper, extract:\n"
                "1. The core problem the paper addresses\n"
                "2. The novelty/contribution of the paper\n"
                "3. How that novelty solves the problem\n\n"
                f"Problem-related excerpts:\n{problem}\n\n"
                f"Novelty-related excerpts:\n{novelty}\n\n"
                "Be concise and structured."
            ))])
            analyses.append(response.content)
            time.sleep(1)
        except Exception as e:
            analyses.append(f"Failed to analyze {pdf_path}: {str(e)}")
    return {"analyses": analyses}


def find_gaps(state: dict) -> dict:
    all_gaps = []
    for i, analysis in enumerate(state["analyses"]):
        try:
            if analysis.startswith("Failed"):
                all_gaps.append(f"Paper {i+1} Gaps: Skipped due to analysis failure.")
                continue
            response = llm.invoke([HumanMessage(content=(
                "You are a research gap identifier.\n"
                "Given this paper analysis, identify:\n"
                "1. What problems remain unsolved\n"
                "2. What limitations exist in the proposed solution\n"
                "3. What future work is explicitly or implicitly needed\n\n"
                f"Analysis:\n{analysis}\n\n"
                "Be concise and structured."
            ))])
            all_gaps.append(f"Paper {i+1} Gaps:\n{response.content}")
            time.sleep(1)
        except Exception as e:
            all_gaps.append(f"Paper {i+1} Gaps: Failed - {str(e)}")
    return {"gaps": all_gaps}


def connect_papers(state: dict) -> dict:
    try:
        analyses = "\n\n".join([f"Paper {i+1} Analysis:\n{a}" for i, a in enumerate(state["analyses"])])
        gaps = "\n\n".join(state["gaps"])
        response = llm.invoke([HumanMessage(content=(
            "You are a research connection expert.\n"
            "Given these analyses and gaps from multiple papers in the same domain:\n\n"
            f"{analyses}\n\n"
            f"{gaps}\n\n"
            "Identify:\n"
            "1. Deep conceptual connections between the papers (not surface-level citations)\n"
            "2. How the novelties of each paper relate or build on each other\n"
            "3. How the gaps across papers compound or relate to each other\n"
            "4. 3-4 substantive unresolved research gaps that emerge from combining all papers\n\n"
            "Be thorough and specific."
        ))])
        return {"connections": response.content}
    except Exception as e:
        return {"connections": f"Connection analysis failed: {str(e)}"}


def synthesize_gaps(state: dict) -> dict:
    try:
        connections = state["connections"]
        response = llm.invoke([HumanMessage(content=(
            "You are a senior research scientist tasked with identifying genuine, substantive research gaps.\n\n"
            f"Based on this cross-paper analysis:\n{connections}\n\n"
            "Your task:\n"
            "- Identify ONLY real, genuine research gaps that are clearly evidenced by the papers\n"
            "- Do NOT fabricate or force gaps that are not strongly supported by the analysis\n"
            "- For each genuine gap found, provide:\n"
            "  1. The gap itself (specific, not vague)\n"
            "  2. Which paper(s) evidence this gap and how\n"
            "  3. Why this gap is still unresolved today\n"
            "  4. A concrete, novel solution direction (only if one is genuinely derivable)\n\n"
            "If fewer than 3 genuine gaps exist, report only what is real.\n"
            "If no novel solution is derivable for a gap, explicitly say so instead of fabricating one.\n\n"
            "Be an honest scientist, not a gap generator."
        ))])
        return {"synthesis": response.content}
    except Exception as e:
        return {"synthesis": f"Synthesis failed: {str(e)}"}