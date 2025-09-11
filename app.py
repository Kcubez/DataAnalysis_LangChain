import streamlit as st
import pandas as pd
import os
import io
import matplotlib.pyplot as plt
import time
import html

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# Custom prompt prefix to ensure the agent uses the full dataframe
PANDAS_AGENT_PREFIX = """
This is a Python environment with pandas installed. You have access to a pandas DataFrame named `df` which contains the FULL dataset that the user has uploaded. You are seeing only the first few rows of the dataframe in this prompt, but you MUST operate on the entire dataframe `df` in your code. Do NOT recreate the dataframe from the head data you see here. All your code should use the `df` variable directly.

CRITICAL RULE: When the user asks for multiple pieces of information (e.g., "total sales and total profit"), your response from the python tool MUST be a single block of text containing all the requested information, with each piece of information on a new line.

Your final answer MUST be the complete, verbatim, multi-line output of your python code. Do not summarize it, rephrase it, or pick only one part of it. Return the entire text block as the final answer.
When you have the final result from the python tool, you MUST use the "Final Answer:" prefix. The text following this prefix MUST be the complete, verbatim, multi-line output from your python code. Do not summarize, rephrase, or select only one part of the output.

Example of a correct final response:
Final Answer: Total Units Sold: 12345
Total Revenue: $678,910.11
Total Profit: $112,131.41
"""

# Use st.cache_data to prevent re-reading the file on every interaction
@st.cache_data
def load_data(file_upload):
    """Reads an uploaded file and returns a pandas DataFrame."""
    try:
        file_extension = os.path.splitext(file_upload.name)[1].lower()
        file_content = file_upload.getvalue()
        if file_extension == '.csv':
            return pd.read_csv(io.BytesIO(file_content))
        elif file_extension == '.xlsx':
            return pd.read_excel(io.BytesIO(file_content))
    except UnicodeDecodeError:
        # Fallback for common encoding issues
        st.warning("File could not be read with standard UTF-8 encoding. Trying with 'latin1'...")
        file_content = file_upload.getvalue() # Re-read bytes for the new reader
        return pd.read_csv(io.BytesIO(file_content), encoding='latin1')

### Streamlit UI

st.set_page_config(page_title="AI Spreadsheet Analyst", layout="wide")
st.title("📊 AI Spreadsheet Analyst")

# --- Custom CSS for a Professional & Theme-Aware UI ---
st.markdown("""
<style>
    /* --- General & Light Mode --- */
    .stApp {
        background-color: var(--background-color);
    }

    /* --- Professional Button Styling --- */
    /* Unifying button styles for "Get Answer", "Save", and "Browse files" */
    div[data-testid="stVerticalBlock"] .stButton > button,
    [data-testid="stFormSubmitButton"] button,
    [data-testid="stFileUploader"] button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease-in-out;
        border: none;
        padding: 10px 24px;
        height: 40px; /* Set a fixed height for alignment */
        background-color: #007bff !important; /* A professional blue */
        color: white !important; /* Use !important to override file uploader's default text color */
    }

    div[data-testid="stVerticalBlock"] .stButton > button:hover,
    [data-testid="stFormSubmitButton"] button:hover,
    [data-testid="stFileUploader"] button:hover {
        background-color: #0069d9 !important; /* A slightly darker blue on hover */
        color: white !important;
    }

    /* Adjust API Key input to match button height */
    [data-testid="stForm"] [data-testid="stTextInput"] input {
        height: 40px !important; /* Match button height */
        box-sizing: border-box !important; /* Include padding in height calculation */
    }

    /* Reset style for the file uploader's clear button ('x') */
    [data-testid="stFileUploader"] button[aria-label^="Remove"] {
        background-color: transparent !important;
        color: inherit !important;
        border: none !important;
        padding: 0 !important; /* Reset padding for the small icon button */
    }

    [data-testid="stFileUploader"] button[aria-label^="Remove"]:hover {
        background-color: transparent !important;
        color: var(--primary-color) !important; /* Add a subtle hover effect */
    }

    /* Spacing for Step headers */
    .step-header {
        margin-top: 2.5rem; /* A more conventional spacing value */
    }

    /* --- Report Styling (Theme-Aware) --- */
    .report-header {
        background-color: #2c3e50; padding: 20px; border-radius: 10px; margin: 20px 0;
        text-align: center; box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .report-header h2 { color: white; margin: 0; font-weight: 600; letter-spacing: 1px; }

    .report-metadata, .data-glance-card, .smart-insight {
        background-color: var(--secondary-background-color);
        border: 1px solid #e1e5e8;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border-radius: 10px;
        padding: 20px;
        margin: 20px 0;
        color: var(--text-color);
    }

    .report-metadata .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; }
    .report-metadata .query { margin-top: 15px; padding-top: 15px; border-top: 1px solid #dee2e6; }

    .key-findings-container {
        background-color: #e8f5e9; border-left: 5px solid #28a745;
        padding: 25px; border-radius: 8px; margin: 20px 0;
    }
    .key-findings-container .content-code {
        font-family: 'Consolas', 'Courier New', monospace; font-size: 18px;
        line-height: 1.6; color: #155724;
        white-space: pre-wrap; word-wrap: break-word;
    }
    .key-findings-container .content-text {
        font-size: 20px; line-height: 1.7; color: #155724;
    }

    .data-glance-card { text-align: center; height: 100%; }
    .data-glance-card p { font-size: 16px; color: var(--text-color); opacity: 0.7; margin: 0 0 5px 0; }
    .data-glance-card h3 { margin: 0; font-size: 28px; color: var(--text-color); }

    .smart-insight h4 { margin: 0 0 10px 0; color: var(--text-color); }
    .smart-insight p { margin: 0; font-size: 16px; color: var(--text-color); }
    .insight-green { border-left: 4px solid #2ecc71; }
    .insight-blue { border-left: 4px solid #3498db; }
    .insight-red { border-left: 4px solid #e74c3c; }

    .report-footer {
        text-align: center;
        padding: 10px 0;
        margin-top: 25px;
        border-top: 1px solid #e1e5e8;
        color: var(--text-color);
        opacity: 0.6;
    }

    /* --- Dark Mode Overrides --- */
    [data-theme="dark"] .report-metadata,
    [data-theme="dark"] .data-glance-card,
    [data-theme="dark"] .smart-insight,
    [data-theme="dark"] .report-footer {
        border: 1px solid #3d3f4b;
    }
    [data-theme="dark"] .report-metadata .query {
        border-top: 1px solid #3d3f4b;
    }

    [data-theme="dark"] .key-findings-container {
        background-color: #0b2d15;
    }
    [data-theme="dark"] .key-findings-container .content-code,
    [data-theme="dark"] .key-findings-container .content-text {
        color: #a3e9b8;
    }

    /* --- Mobile View Adjustments --- */
    @media (max-width: 768px) {
        .report-metadata .grid {
            grid-template-columns: 1fr; /* Stack grid items into a single column */
            gap: 10px; /* Adjust gap for stacked layout */
        }
        /* Reduce vertical space between stacked "Dataset at a Glance" cards on mobile */
        .data-glance-card {
            margin: 10px 0;
        }
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if 'df' not in st.session_state:
    st.session_state.df = None
if 'api_key' not in st.session_state:
    st.session_state.api_key = None

# --- Configuration Expander for API Key ---
with st.expander("🔑 Configuration: Enter Your API Key", expanded=not st.session_state.api_key):
    # Display a status message if the key is already set
    if st.session_state.api_key:
        st.info("An API key is already configured. Enter a new key to replace it, or submit an empty field to clear it.")

    with st.form(key='api_key_form'):
        # Use columns for responsive layout. They will be side-by-side on desktop
        # and stacked vertically on mobile.
        col1, col2 = st.columns([5, 1])

        with col1:
            api_key_input = st.text_input(
                "Google API Key",
                type="password",
                placeholder="Enter your Google API Key...",
                label_visibility="collapsed"
            )

        with col2:
            submitted = st.form_submit_button(label="Save", use_container_width=True)

        if submitted:
            st.session_state.api_key = api_key_input or None
            if api_key_input:
                st.success("✓ API Key saved!")
            else:
                st.warning("API Key cleared.")

            time.sleep(1) # Give user time to see message
            st.rerun()

# --- Main App Logic ---
if not st.session_state.api_key:
    st.info("Please enter your Google API Key in the configuration section above to start the analysis.")
    st.stop()

st.markdown("<div class='step-header'><h3>Step 1: Upload Your Spreadsheet</h3></div>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload a CSV or Excel file to begin.", type=["csv", "xlsx"], label_visibility="collapsed")

if uploaded_file is not None:
    try:
        st.session_state.df = load_data(uploaded_file)
        st.success("File loaded successfully!")
        st.markdown("#### Data Preview (First 5 Rows)")
        st.dataframe(st.session_state.df.head())
    except Exception as e:
        st.error(f"An unexpected error occurred while loading the file: {e}")

    # Text input for the report query
    st.markdown("<div class='step-header'><h3>Step 2: Ask Your Question</h3></div>", unsafe_allow_html=True)
    user_query = st.text_area(
        "Enter your question about the data:",
        placeholder="e.g., 'What is the average age?' or 'Make a bar chart of total profit by country.'",
        height=100
    )

    # Button to generate the report
    if st.button("Get Answer", use_container_width=True):
        if user_query and st.session_state.df is not None:
            with st.spinner("The AI is analyzing your data..."):
                try:
                    # 1. Define the LLM with reduced temperature for more deterministic responses
                    llm = ChatGoogleGenerativeAI(
                        model="gemini-2.5-flash",
                        google_api_key=st.session_state.api_key,
                        temperature=0, 
                        max_tokens=4096,
                        timeout=60,
                        safety_settings={
                            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                        }
                    )

                    # 2. Create the Pandas DataFrame Agent with the custom prefix
                    agent = create_pandas_dataframe_agent(
                        llm,
                        st.session_state.df.copy(),
                        prefix=PANDAS_AGENT_PREFIX,
                        verbose=True,
                        allow_dangerous_code=True,
                        max_iterations=5,  # Increased but still limited
                        return_intermediate_steps=True,  # Force return of intermediate steps
                    )

                    # 3. Add special instructions for plotting
                    query_with_instructions = user_query
                    if any(keyword in user_query.lower() for keyword in ["chart", "plot", "graph", "bar", "pie", "line"]):
                        query_with_instructions += (
                            "\n\nCHARTING INSTRUCTIONS: The Y-axis must represent the actual, absolute data values (like sum of profit), not percentages or ratios. Ensure the chart is not a '100% stacked' or 'normalized' chart."
                        )
                    
                    query_with_instructions += (
                        "\n\nIMPORTANT: If you create a plot, you MUST save it to a file named 'plot.png' and then just say 'I have created the plot.' Do not use plt.show(). Execute your calculation ONLY ONCE and provide the Final Answer immediately. Do NOT repeat calculations."
                    )

                    # 4. Invoke the agent with timeout handling
                    try:
                        start_time = time.time()
                        answer = agent.invoke(query_with_instructions)
                        output = answer.get('output', 'No output found.')

                        # Debug: Print what we got from the agent
                        print(f"DEBUG: Agent raw output: {output}")
                        print(f"DEBUG: Full answer structure: {answer}")

                        # --- REVISED FALLBACK & ERROR HANDLING ---
                        # The agent's 'output' can be a summary. Check the last observation from
                        # intermediate_steps to see if it's a more complete, multi-line answer.
                        try:
                            intermediate_steps = answer.get('intermediate_steps', [])
                            if intermediate_steps:
                                # The observation is the second element of the (action, observation) tuple
                                last_observation = str(intermediate_steps[-1][1]).strip()

                                # Heuristic: If the last observation has multiple lines and the final
                                # output is just a single line, the agent likely summarized it.
                                # Use the fuller observation, as long as it's not an error.
                                if ('\n' in last_observation and '\n' not in output and
                                    "error" not in last_observation.lower() and
                                    "exception" not in last_observation.lower()):
                                    print("DEBUG: Final answer was single-line; using multi-line observation from last step instead.")
                                    output = last_observation
                        except Exception as e:
                            # This is a fallback, so if it fails, we just proceed with the original output.
                            print(f"DEBUG: Could not process intermediate steps for fallback: {e}")

                    except Exception as parsing_error:
                        error_message = str(parsing_error)
                        
                        # Handle rate limiting specifically
                        if "ResourceExhausted" in error_message or "429" in error_message:
                            output = "⚠️ API rate limit exceeded. Please wait a moment and try again. The free tier allows 10 requests per minute."
                        elif "Final Answer:" in error_message:
                            # Extract the final answer from the error message
                            final_answer_start = error_message.find("Final Answer:")
                            if final_answer_start != -1:
                                output = error_message[final_answer_start + len("Final Answer:"):].strip()
                                # Clean up the output by removing troubleshooting text
                                if "For troubleshooting" in output:
                                    output = output.split("For troubleshooting")[0].strip()
                            else:
                                output = f"Parsing error occurred, but here's what I found: {error_message}"
                        else:
                            raise parsing_error

                    # Handle cases where the agent stops or fails to produce a meaningful output
                    if (not output or output == 'No output found.' or
                        'Agent stopped due to iteration limit' in str(answer) or
                        'Agent stopped due to time limit' in str(answer)):
                        output = "The AI was unable to provide a direct answer. This might be due to the complexity of the question or an internal limit. Please try asking a simpler question."



                    # 5. Check if the plot file was created and display it
                    if os.path.exists('plot.png'):
                        st.image('plot.png', caption="Generated Chart",  width='stretch')
                        os.remove('plot.png') # Clean up the file

                    # Professional & Clean Report Format
                    if output:
                        st.markdown('<div class="report-header"><h2>Analysis Report</h2></div>', unsafe_allow_html=True)

                        # Report Metadata - Clean & Simple
                        current_time = time.strftime("%B %d, %Y at %I:%M %p")
                        st.markdown(f"""
                        <div class="report-metadata">
                            <div class="grid">
                                <div><strong>📅 Generated:</strong> {current_time}</div>
                                <div><strong>📁 Dataset:</strong> {html.escape(uploaded_file.name)}</div>
                            </div>
                            <div class="query">
                                <strong>❓ Query:</strong> <em>"{html.escape(user_query)}"</em>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        # Main Answer - Formatted for clarity
                        st.markdown("### 💡 Key Findings")
                        safe_output = html.escape(output)

                        # Check for multi-line or numerical output
                        if '\n' in output or any(char.isdigit() for char in output):
                            st.markdown(f'<div class="key-findings-container"><div class="content-code">{safe_output}</div></div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="key-findings-container"><div class="content-text">{safe_output}</div></div>', unsafe_allow_html=True)

                        # Quick Data Overview - Compact
                        st.markdown("### 📋 Dataset at a Glance")
                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown(f'<div class="data-glance-card"><p>Rows</p><h3>{len(st.session_state.df):,}</h3></div>', unsafe_allow_html=True)

                        with col2:
                            st.markdown(f'<div class="data-glance-card"><p>Columns</p><h3>{len(st.session_state.df.columns)}</h3></div>', unsafe_allow_html=True)

                        # Smart Insights - Context Aware & Brief
                        if any(keyword in user_query.lower() for keyword in ["top", "best", "highest", "maximum", "largest"]):
                            st.markdown("""
                            <div class="smart-insight insight-green">
                                <h4>💡 Smart Insight</h4>
                                <p>This analysis focuses on <strong>top-performing</strong> metrics in your dataset.</p>
                            </div>
                            """, unsafe_allow_html=True)
                        elif any(keyword in user_query.lower() for keyword in ["average", "mean", "median"]):
                            st.markdown("""
                            <div class="smart-insight insight-blue">
                                <h4>💡 Smart Insight</h4>
                                <p>This report includes <strong>statistical analysis</strong> (e.g., mean, median) of your data.</p>
                            </div>
                            """, unsafe_allow_html=True)
                        elif any(keyword in user_query.lower() for keyword in ["chart", "plot", "graph", "bar", "pie", "line"]):
                            st.markdown("""
                            <div class="smart-insight insight-red">
                                <h4>💡 Smart Insight</h4>
                                <p>A <strong>visualization</strong> has been generated to illustrate patterns in your data.</p>
                            </div>
                            """, unsafe_allow_html=True)

                        # Clean Footer
                        st.markdown('<div class="report-footer"><small>Report generated by AI Spreadsheet Analyst</small></div>', unsafe_allow_html=True)

                except Exception as e:
                    error_message = str(e)
                    # Check for invalid API key error and provide a user-friendly message
                    if "API key not valid" in error_message or "API_KEY_INVALID" in error_message:
                        st.error("The Google API Key is invalid. Please enter a valid key in the configuration section above and try again.")
                    else:
                        st.error(f"An error occurred: {e}")
        else:
            st.warning("Please upload a file and enter a query.")