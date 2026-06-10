import streamlit as st
import sys
import os
import traceback
import pandas as pd

# ==================================================
# Project Path
# ==================================================

project_root = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

if project_root not in sys.path:
    sys.path.append(project_root)

from src.pipeline.rag_pipeline import RAGPipeline

# ==================================================
# Page Config
# ==================================================

st.set_page_config(
    page_title="IFC RAG System",
    page_icon="📚",
    layout="wide"
)

# ==================================================
# Title
# ==================================================

st.title("📚 IFC Annual Report 2024 RAG System")

st.markdown(
    """
### Supported Features

✅ Dense Retrieval

✅ BM25 Retrieval

✅ Hybrid Retrieval

✅ Multi-Hop Retrieval

✅ Table Retrieval

✅ Image Retrieval

✅ Cross Encoder Re-ranking

✅ LLM Re-ranking
"""
)

# ==================================================
# Load Pipeline
# ==================================================

@st.cache_resource
def load_pipeline():

    pdf_path = (
        "data/pdf/IFC_Annual_Report_2024.pdf"
    )

    return RAGPipeline(pdf_path)

try:

    pipeline = load_pipeline()

    st.success(
        "✅ Pipeline Loaded Successfully"
    )

except Exception as e:

    st.error(
        f"Pipeline Loading Failed: {e}"
    )

    st.code(
        traceback.format_exc()
    )

    st.stop()

# ==================================================
# Sidebar
# ==================================================

st.sidebar.title("⚙️ Settings")

retrieval_mode = st.sidebar.selectbox(
    "Retrieval Mode",
    [
        "Dense",
        "BM25",
        "Hybrid",
        "Multi-Hop",
        "Multimodal"
    ]
)

page_start = st.sidebar.number_input(
    "Start Page",
    min_value=1,
    value=1
)

page_end = st.sidebar.number_input(
    "End Page",
    min_value=1,
    value=300
)

st.sidebar.markdown("---")

st.sidebar.write(
    f"Total Chunks: {len(pipeline.all_chunks)}"
)

# ==================================================
# Question Input
# ==================================================

question = st.text_input(
    "Enter your question"
)

# ==================================================
# Ask Button
# ==================================================

if st.button("Ask"):

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        try:

            with st.spinner(
                "Searching..."
            ):

                answer, docs = pipeline.ask(
                    question=question,
                    retrieval_mode=retrieval_mode,
                    page_start=page_start,
                    page_end=page_end
                )

            # ======================================
            # Answer
            # ======================================

            st.subheader("Answer")

            st.write(answer)

            st.success(
                f"Retrieval Mode Used: {retrieval_mode}"
            )

            pages = sorted(
                {
                    doc.get("page", 0)
                    for doc in docs
                }
            )

            st.write(
                f"📄 Source Pages: {pages}"
            )

            if retrieval_mode == "Multimodal" and docs:

                st.subheader(
                    "Retrieved Visual Evidence"
                )

                for doc in docs:

                    if not doc.get("path"):
                        continue

                    st.image(
                        doc["path"],
                        caption=(
                            f"Page {doc.get('page', 'N/A')} "
                            f"| Patch {doc.get('patch', 'N/A')}"
                        )
                    )

                    st.write(
                        f"Similarity Score: "
                        f"{doc.get('score', 0.0):.4f}"
                    )

            # ======================================
            # Retrieved Chunks
            # ======================================

            st.subheader(
                "Retrieved Chunks"
            )

            for i, doc in enumerate(
                docs,
                start=1
            ):

                page = doc.get(
                    "page",
                    "N/A"
                )

                section = doc.get(
                    "section",
                    "Unknown"
                )

                score = doc.get(
                    "score",
                    0.0
                )

                llm_score = doc.get(
                    "llm_score",
                    0.0
                )

                content_type = doc.get(
                    "content_type",
                    "text"
                )

                with st.expander(
                    f"Chunk {i} | Page {page} | {content_type}"
                ):

                    st.write(
                        f"Section: {section}"
                    )

                    st.write(
                        f"Score: {score:.4f}"
                    )

                    st.write(
                        f"LLM Score: {llm_score:.4f}"
                    )

                    st.write(
                        f"Content Type: {content_type}"
                    )

                    # ==================================
                    # IMAGE CHUNKS
                    # ==================================

                    if content_type == "image":

                        image_path = doc.get(
                            "source",
                            ""
                        )

                        if (
                            image_path
                            and os.path.exists(
                                image_path
                            )
                        ):

                            st.image(
                                image_path,
                                caption=f"Page {page}"
                            )

                        st.markdown(
                            "### Image Description"
                        )

                        st.write(
                            doc.get(
                                "content",
                                ""
                            )
                        )

                    # ==================================
                    # TABLE CHUNKS
                    # ==================================

                    elif content_type == "table":

                        st.markdown(
                            "### Retrieved Table"
                        )

                        table_text = doc.get(
                            "content",
                            ""
                        )

                        try:

                            rows = []

                            for row in table_text.split("\n"):

                                cells = [
                                    cell.strip()
                                    for cell in row.split("|")
                                ]

                                rows.append(cells)

                            if len(rows) > 1:

                                header = rows[0]

                                data = rows[1:]

                                df = pd.DataFrame(
                                    data,
                                    columns=header
                                )

                                st.dataframe(
                                    df,
                                    use_container_width=True
                                )

                            else:

                                st.text(
                                    table_text
                                )

                        except Exception:

                            st.text(
                                table_text
                            )

                    # ==================================
                    # TEXT CHUNKS
                    # ==================================

                    else:

                        st.markdown(
                            "### Retrieved Text"
                        )

                        st.write(
                            doc.get(
                                "content",
                                ""
                            )
                        )

                    # ==================================
                    # SOURCE
                    # ==================================

                    if "source" in doc:

                        st.caption(
                            f"Source: {doc['source']}"
                        )

        except Exception as e:

            st.error(
                f"Error: {str(e)}"
            )

            st.code(
                traceback.format_exc()
            )