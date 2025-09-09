import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import os
import io
import matplotlib.pyplot as plt
import time
import html

# LCEL and LLM imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# Load environment variables for API keys
load_dotenv()

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

### Streamlit UI

st.set_page_config(page_title="AI Spreadsheet Analyst", layout="wide")
st.title("📊 AI Spreadsheet Analyst")
st.markdown("Upload a CSV or Excel file and ask questions about your data. You can also ask for charts!")

# Session state to store the dataframe
if 'df' not in st.session_state:
    st.session_state.df = None

# File uploader
uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Use st.cache_data to prevent re-reading the file on every interaction
    @st.cache_data
    def load_data(file):
        file_extension = os.path.splitext(file.name)[1].lower()
        if file_extension == '.csv':
            try:
                return pd.read_csv(file)
            except UnicodeDecodeError:
                return pd.read_csv(file, encoding='latin1')
        elif file_extension == '.xlsx':
            return pd.read_excel(file)
        else:
            raise ValueError("Unsupported file format. Please upload a CSV or XLSX file.")

    try:
        st.session_state.df = load_data(uploaded_file)
        st.success("File loaded successfully!")
        st.subheader("First 5 rows of the data")
        st.dataframe(st.session_state.df.head())
    except Exception as e:
        st.error(f"An unexpected error occurred while loading the file: {e}")

    # Text input for the report query
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
                        st.session_state.df,
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
                        st.image('plot.png', caption="Generated Chart", use_column_width=True)
                        os.remove('plot.png') # Clean up the file

                    # Professional & Clean Report Format
                    if output:
                        # Professional Header
                        st.markdown("""
                        <div style="background-color: #2c3e50;
                                    padding: 20px; border-radius: 10px; margin: 20px 0; text-align: center; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
                            <h2 style="color: white; margin: 0; font-weight: 600; letter-spacing: 1px;">Analysis Report</h2>
                        </div>
                        """, unsafe_allow_html=True)

                        # Report Metadata - Clean & Simple
                        current_time = time.strftime("%B %d, %Y at %I:%M %p")
                        st.markdown(f"""
                        <div style="background-color: #ffffff; border: 1px solid #e1e5e8; box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                                    border-radius: 10px; padding: 20px; margin: 20px 0; color: #34495e;">
                            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
                                <div><strong>📅 Generated:</strong> {current_time}</div>
                                <div><strong>📁 Dataset:</strong> {uploaded_file.name}</div>
                            </div>
                            <div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid #dee2e6;">
                                <strong>❓ Query:</strong> <em>"{user_query}"</em>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        # Main Answer - Formatted for clarity
                        st.markdown("### 💡 Key Findings")

                        # Pre-format the output to handle newlines for HTML and escape characters
                        formatted_output = html.escape(output).replace('\n', '<br>')

                        # Check for multi-line or numerical output
                        if '\n' in output or any(char.isdigit() for char in output):
                            # For numerical or multi-line results - clean and structured
                            st.markdown(f"""
                            <div style="background-color: #e8f5e9; border-left: 5px solid #28a745;
                                        padding: 25px; border-radius: 8px; margin: 20px 0;">
                                <div style="font-family: 'Consolas', 'Courier New', monospace; font-size: 18px;
                                           line-height: 1.6; color: #155724;">
                                    {formatted_output}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            # For qualitative/text results - clean and readable
                            st.markdown(f"""
                            <div style="background-color: #e8f5e9; border-left: 5px solid #28a745;
                                        padding: 25px; border-radius: 8px; margin: 20px 0;">
                                <div style="font-size: 20px; line-height: 1.7; color: #155724;">
                                    {formatted_output}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                        # Quick Data Overview - Compact
                        st.markdown("### 📋 Dataset at a Glance")
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.markdown(f"""
                            <div style="background-color: #ffffff; border: 1px solid #e1e5e8; box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                                        border-radius: 10px; padding: 20px; text-align: center;">
                                <p style="font-size: 16px; color: #7f8c8d; margin: 0 0 5px 0;">Rows</p>
                                <h3 style="margin: 0; color: #2c3e50; font-size: 28px;">{len(st.session_state.df):,}</h3>
                            </div>
                            """, unsafe_allow_html=True)

                        with col2:
                            st.markdown(f"""
                            <div style="background-color: #ffffff; border: 1px solid #e1e5e8; box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                                        border-radius: 10px; padding: 20px; text-align: center;">
                                <p style="font-size: 16px; color: #7f8c8d; margin: 0 0 5px 0;">Columns</p>
                                <h3 style="margin: 0; color: #2c3e50; font-size: 28px;">{len(st.session_state.df.columns)}</h3>
                            </div>
                            """, unsafe_allow_html=True)

                        with col3:
                            numeric_cols = len(st.session_state.df.select_dtypes(include=['number']).columns)
                            st.markdown(f"""
                            <div style="background-color: #ffffff; border: 1px solid #e1e5e8; box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                                        border-radius: 10px; padding: 20px; text-align: center;">
                                <p style="font-size: 16px; color: #7f8c8d; margin: 0 0 5px 0;">Numeric Columns</p>
                                <h3 style="margin: 0; color: #2c3e50; font-size: 28px;">{numeric_cols}</h3>
                            </div>
                            """, unsafe_allow_html=True)

                        # Smart Insights - Context Aware & Brief
                        if any(keyword in user_query.lower() for keyword in ["top", "best", "highest", "maximum", "largest"]):
                            st.markdown("""
                            <div style="background-color: #f7f9f9; border-left: 4px solid #2ecc71;
                                        padding: 20px; border-radius: 8px; margin: 20px 0;">
                                <h4 style="margin: 0 0 10px 0; color: #2c3e50;">💡 Smart Insight</h4>
                                <p style="margin: 0; color: #34495e; font-size: 16px;">
                                    This analysis focuses on <strong>top-performing</strong> metrics in your dataset.
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                        elif any(keyword in user_query.lower() for keyword in ["average", "mean", "median"]):
                            st.markdown("""
                            <div style="background-color: #f7f9f9; border-left: 4px solid #3498db;
                                        padding: 20px; border-radius: 8px; margin: 20px 0;">
                                <h4 style="margin: 0 0 10px 0; color: #2c3e50;">💡 Smart Insight</h4>
                                <p style="margin: 0; color: #34495e; font-size: 16px;">
                                    This report includes <strong>statistical analysis</strong> (e.g., mean, median) of your data.
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                        elif any(keyword in user_query.lower() for keyword in ["chart", "plot", "graph", "bar", "pie", "line"]):
                            st.markdown("""
                            <div style="background-color: #f7f9f9; border-left: 4px solid #e74c3c;
                                        padding: 20px; border-radius: 8px; margin: 20px 0;">
                                <h4 style="margin: 0 0 10px 0; color: #2c3e50;">💡 Smart Insight</h4>
                                <p style="margin: 0; color: #34495e; font-size: 16px;">
                                    A <strong>visualization</strong> has been generated to illustrate patterns in your data.
                                </p>
                            </div>
                            """, unsafe_allow_html=True)

                        # Clean Footer
                        st.markdown("""
                        <div style="text-align: center; padding: 20px; margin-top: 40px;
                                    border-top: 1px solid #e1e5e8; color: #95a5a6;">
                            <small>Report generated by AI Spreadsheet Analyst</small>
                        </div>
                        """, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"An error occurred: {e}")
        else:
            st.warning("Please upload a file and enter a query.")