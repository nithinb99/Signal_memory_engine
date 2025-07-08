import streamlit as st
import requests

# Basic Streamlit UI for Signal Memory RAG backend

def main():
    st.title("Signal Memory RAG Interface")
    st.sidebar.header("Settings")

    backend_url = st.sidebar.text_input("Backend URL", "http://localhost:8000")
    mode = st.sidebar.selectbox("Mode", ["Single-Agent", "Multi-Agent"])
    k = st.sidebar.slider("Number of chunks (k)", min_value=1, max_value=10, value=3)

    st.header("Enter your query")
    query = st.text_area("", height=100)

    if st.button("Submit"):
        if not query.strip():
            st.warning("Please enter a query.")
            return

        endpoint = "/query" if mode == "Single-Agent" else "/multi_query"
        url = f"{backend_url}{endpoint}"
        payload = {"query": query, "k": k}

        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            st.error(f"Request failed: {e}")
            return

        # Display results
        if mode == "Single-Agent":
            st.subheader("Answer")
            st.write(data.get("answer", ""))

            st.subheader("Memory Chunks & Scores")
            for chunk in data.get("chunks", []):
                st.markdown(f"- **{chunk['score']:.3f}**: {chunk['content']}")

            st.subheader("Flag & Suggestion")
            st.markdown(f"**Flag:** {data.get('flag', '')}")
            st.markdown(f"**Suggestion:** {data.get('suggestion', '')}")

        else:
            st.subheader("Multi-Agent Responses")
            agents = data.get("agents", {})
            for role, result in agents.items():
                st.markdown(f"### {role}")
                st.markdown(f"**Answer:** {result.get('answer', '')}")

                st.markdown("**Chunks & Scores:**")
                for chunk in result.get('chunks', []):
                    st.markdown(f"- **{chunk['score']:.3f}**: {chunk['content']}")

                st.markdown(f"**Flag:** {result.get('flag', '')}")
                st.markdown(f"**Suggestion:** {result.get('suggestion', '')}")

if __name__ == "__main__":
    main()