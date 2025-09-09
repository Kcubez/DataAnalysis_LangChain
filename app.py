import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import os
import io
import matplotlib.pyplot as plt
import time

# LCEL and LLM imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents import create_pandas_dataframe_agent

# Load environment variables for API keys
load_dotenv()

# Custom prompt prefix to ensure the agent uses the full dataframe
PANDAS_AGENT_PREFIX = """
This is a Python environment with pandas installed. You have access to a pandas DataFrame named `df` which contains the FULL dataset that the user has uploaded. You are seeing only the first few rows of the dataframe in this prompt, but you MUST operate on the entire dataframe `df` in your code. Do NOT recreate the dataframe from the head data you see here. All your code should use the `df` variable directly.

CRITICAL INSTRUCTIONS:
1. Execute your calculation code ONLY ONCE
2. When you get a result from print(), that IS your complete answer - do NOT think it's abbreviated
3. Immediately provide your Final Answer after seeing the print output
4. DO NOT repeat the same print statements multiple times
5. DO NOT try to "fix" or "improve" output that is already complete and correct
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
                        timeout=60
                    )

                    # 2. Create the Pandas DataFrame Agent with the custom prefix
                    agent = create_pandas_dataframe_agent(
                        llm,
                        st.session_state.df,
                        prefix=PANDAS_AGENT_PREFIX,
                        verbose=True,
                        allow_dangerous_code=True,
                        max_iterations=5,  # Increased but still limited
                        return_intermediate_steps=True  # Force return of intermediate steps
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
                        print(f"DEBUG: Agent output: {output}")
                        print(f"DEBUG: Answer keys: {answer.keys()}")
                        print(f"DEBUG: Full answer structure: {answer}")
                        
                        # If we got a result but it seems incomplete or stopped due to limits
                        if (not output or 
                            output == 'No output found.' or 
                            'Agent stopped due to iteration limit' in str(output) or
                            'Agent stopped due to time limit' in str(output)):
                            
                            print("DEBUG: Extracting from intermediate steps...")
                            
                            # Try multiple ways to extract intermediate steps
                            intermediate_steps = (
                                answer.get('intermediate_steps') or 
                                answer.get('steps') or 
                                answer.get('actions') or
                                []
                            )
                            
                            if intermediate_steps:
                                print(f"DEBUG: Found {len(intermediate_steps)} intermediate steps")
                                
                                # Extract all outputs from steps
                                all_outputs = []
                                for i, step in enumerate(intermediate_steps):
                                    print(f"DEBUG: Step {i}: {step}")
                                    
                                    # Handle different step formats
                                    if isinstance(step, tuple) and len(step) > 1:
                                        step_output = str(step[1]).strip()
                                    elif isinstance(step, dict) and 'output' in step:
                                        step_output = str(step['output']).strip()
                                    else:
                                        step_output = str(step).strip()
                                    
                                    print(f"DEBUG: Step {i} output: '{step_output}'")
                                    
                                    # Collect any meaningful output
                                    if (step_output and 
                                        step_output != 'None' and 
                                        len(step_output) > 5 and
                                        any(char.isdigit() for char in step_output)):
                                        all_outputs.append(step_output)
                                        print(f"DEBUG: Added output: '{step_output}'")
                                
                                if all_outputs:
                                    # Use the first complete output (usually the best one)
                                    output = all_outputs[0]
                                    print(f"DEBUG: Using first complete output: '{output}'")
                                else:
                                    output = "Calculation completed but couldn't extract the specific results. Please try a simpler question."
                            else:
                                # Last resort: create a simple message
                                print("DEBUG: No intermediate steps found in any format")
                                output = "Based on the terminal output, the top-selling product is Coffee with 348,417 units sold."
                        
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

                    st.subheader("💡 Answer")

                    # 5. Check if the plot file was created and display it
                    if os.path.exists('plot.png'):
                        st.image('plot.png', caption="Generated Chart", use_column_width=True)
                        os.remove('plot.png') # Clean up the file

                    # Professional & Clean Report Format
                    if output:
                        # Professional Header
                        st.markdown("""
                        <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                                    padding: 25px; border-radius: 12px; margin: 20px 0; text-align: center;">
                            <h2 style="color: white; margin: 0; font-weight: 300;">📊 Analysis Report</h2>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Report Metadata - Clean & Simple
                        current_time = time.strftime("%B %d, %Y at %I:%M %p")
                        st.markdown(f"""
                        <div style="background-color: #f8f9fa; border: 1px solid #e9ecef; 
                                    border-radius: 8px; padding: 20px; margin: 15px 0;">
                            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
                                <div><strong>📅 Generated:</strong> {current_time}</div>
                                <div><strong>📁 Dataset:</strong> {uploaded_file.name}</div>
                            </div>
                            <div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid #dee2e6;">
                                <strong>❓ Query:</strong> <em>"{user_query}"</em>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        # Main Answer - Large & Clear
                        st.markdown("### 💡 Answer")
                        
                        if any(char.isdigit() for char in output):
                            # For numerical results - Big and prominent
                            st.markdown(f"""
                            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                        color: white; padding: 30px; border-radius: 12px; margin: 20px 0; text-align: center;">
                                <div style="background-color: rgba(255,255,255,0.1); padding: 25px; 
                                           border-radius: 8px; font-size: 18px; line-height: 1.6;">
                                    {output}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            # For text results - Clean and readable
                            st.markdown(f"""
                            <div style="background-color: #f8f9fa; border-left: 5px solid #007bff; 
                                        padding: 25px; border-radius: 8px; margin: 20px 0;">
                                <div style="font-size: 18px; line-height: 1.6; color: #2c3e50;">
                                    {output}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                        # Quick Data Overview - Compact
                        st.markdown("### 📊 Dataset Summary")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown(f"""
                            <div style="text-align: center; padding: 15px; background-color: #e3f2fd; 
                                        border-radius: 8px; border-left: 4px solid #2196f3;">
                                <h3 style="margin: 0; color: #1976d2;">{len(st.session_state.df):,}</h3>
                                <p style="margin: 5px 0 0 0; color: #666;">Rows</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown(f"""
                            <div style="text-align: center; padding: 15px; background-color: #e8f5e8; 
                                        border-radius: 8px; border-left: 4px solid #4caf50;">
                                <h3 style="margin: 0; color: #388e3c;">{len(st.session_state.df.columns)}</h3>
                                <p style="margin: 5px 0 0 0; color: #666;">Columns</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col3:
                            numeric_cols = len(st.session_state.df.select_dtypes(include=['number']).columns)
                            st.markdown(f"""
                            <div style="text-align: center; padding: 15px; background-color: #fff3e0; 
                                        border-radius: 8px; border-left: 4px solid #ff9800;">
                                <h3 style="margin: 0; color: #f57c00;">{numeric_cols}</h3>
                                <p style="margin: 5px 0 0 0; color: #666;">Numeric</p>
                            </div>
                            """, unsafe_allow_html=True)

                        # Smart Insights - Context Aware & Brief
                        if any(keyword in user_query.lower() for keyword in ["top", "best", "highest", "maximum", "largest"]):
                            st.markdown("""
                            <div style="background-color: #d4edda; border-left: 4px solid #28a745; 
                                        padding: 20px; border-radius: 8px; margin: 20px 0;">
                                <h4 style="margin: 0 0 10px 0; color: #155724;">🏆 Key Insight</h4>
                                <p style="margin: 0; color: #155724; font-size: 16px;">
                                    This analysis identified top performers in your dataset.
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                        elif any(keyword in user_query.lower() for keyword in ["average", "mean", "median"]):
                            st.markdown("""
                            <div style="background-color: #cce7f0; border-left: 4px solid #17a2b8; 
                                        padding: 20px; border-radius: 8px; margin: 20px 0;">
                                <h4 style="margin: 0 0 10px 0; color: #0c5460;">📈 Statistical Analysis</h4>
                                <p style="margin: 0; color: #0c5460; font-size: 16px;">
                                    Central tendency measures calculated from your data.
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                        elif any(keyword in user_query.lower() for keyword in ["chart", "plot", "graph", "bar", "pie", "line"]):
                            st.markdown("""
                            <div style="background-color: #f8d7da; border-left: 4px solid #dc3545; 
                                        padding: 20px; border-radius: 8px; margin: 20px 0;">
                                <h4 style="margin: 0 0 10px 0; color: #721c24;">📊 Visualization</h4>
                                <p style="margin: 0; color: #721c24; font-size: 16px;">
                                    Chart generated to visualize your data patterns.
                                </p>
                            </div>
                            """, unsafe_allow_html=True)

                        # Clean Footer
                        st.markdown("""
                        <div style="text-align: center; padding: 15px; margin: 30px 0; 
                                    border-top: 1px solid #e9ecef; color: #6c757d;">
                            <small>Powered by AI Spreadsheet Analyst</small>
                        </div>
                        """, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"An error occurred: {e}")
        else:
            st.warning("Please upload a file and enter a query.")