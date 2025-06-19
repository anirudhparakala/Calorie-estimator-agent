import streamlit as st
import google.generativeai as genai
from tavily import TavilyClient
import json
from PIL import Image
import io

# --- Page Configuration ---
st.set_page_config(
    page_title="Intelligent AI Calorie Estimator",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="auto"
)

# --- API Configuration ---
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    TAVILY_API_KEY = st.secrets["TAVILY_API_KEY"]
    genai.configure(api_key=GEMINI_API_KEY)
    tavily = TavilyClient(api_key=TAVILY_API_KEY)
except (FileNotFoundError, KeyError):
    st.error("API keys not found in secrets. Please check your .streamlit/secrets.toml file.", icon="⚠️")
    st.stop()


# --- Tool Definition ---
def perform_web_search(query: str):
    """
    Performs a web search to find nutritional information for specific food items.
    """
    try:
        print(f"Performing search for: {query}")
        results = tavily.search(query=query, search_depth="basic")
        return json.dumps([{"url": obj["url"], "content": obj["content"]} for obj in results['results']])
    except Exception as e:
        print(f"Error during search: {e}")
        return f"Error performing search: {e}"


# --- Model and Tool Initialization ---
my_search_tool = genai.protos.Tool(
    function_declarations=[
        genai.protos.FunctionDeclaration(
            name='perform_web_search',
            description="Performs a web search using the Tavily API to find nutritional information for specific food items, especially branded or restaurant items. Use this to find calorie counts, macronutrient breakdowns (protein, carbs, fat), and average weights or serving sizes. For example: 'calories in Burger King Whopper' or 'average weight of a Walmart Great Value chicken breast'.",
            parameters=genai.protos.Schema(
                type=genai.protos.Type.OBJECT,
                properties={
                    'query': genai.protos.Schema(type=genai.protos.Type.STRING,
                                                 description="The precise search query string.")
                },
                required=['query']
            )
        )
    ]
)

available_tools = {
    "perform_web_search": perform_web_search,
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-pro-latest",
    tools=[my_search_tool]
)


def load_css():
    st.markdown("""<style>/* Your custom CSS can go here */</style>""", unsafe_allow_html=True)


def main():
    load_css()
    st.title("Intelligent AI Calorie Estimator 🧠")

    if "analysis_stage" not in st.session_state:
        st.session_state.analysis_stage = "upload"
    if "uploaded_image_data" not in st.session_state:
        st.session_state.uploaded_image_data = None
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if st.session_state.analysis_stage == "upload":
        st.info("Upload a food photo. The AI will act as your expert estimator.", icon="🧑‍🔬")
        uploaded_file = st.file_uploader("Upload an image of your meal...", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            st.session_state.uploaded_image_data = uploaded_file.getvalue()
            st.session_state.analysis_stage = "analyzing"
            st.rerun()

    if st.session_state.analysis_stage == "analyzing":
        st.image(st.session_state.uploaded_image_data, caption="Your meal.", use_container_width=True)
        if st.button("🔍 Analyze Food"):
            with st.spinner("Performing expert analysis..."):
                # --- FINAL PROMPT WITH EXPLICIT FALLBACK RULE ---
                prompt = """
                // SYSTEM CONSTITUTION: Nutri-AI v3.5

                // 1. CORE IDENTITY: You are "Nutri-AI," an expert visual nutritional estimator.
                // 2. PRIMARY DIRECTIVE: ESTIMATE FIRST. Always make your own visual estimate of quantity first and state it clearly.
                // 3. RULE OF INQUIRY: Ask for simple confirmations. For composite items (shakes, stews), you must ask for ingredients. If you hit a dead end, try asking a more open-ended question.

                // 4. RULE OF TOOL USE (WITH FALLBACK):
                // 4a. If the user mentions a specific brand or restaurant, use the `perform_web_search` tool to find specific data.
                // 4b. If the user corrects your findings, try to perform a new search with the more specific information.
                // 4c. NEW - FALLBACK RULE: If the web search fails to find specific nutritional data for a brand/restaurant, you MUST inform the user that you couldn't find specific info, and then IMMEDIATELY provide an estimate based on your general knowledge of that food type (e.g., "I couldn't find the exact details for that restaurant's biryani, but a typical plate of chicken biryani has about...").

                // 5. CONVERSATIONAL BOUNDARY: Your role is ONLY to gather information. DO NOT provide calorie counts or final calculations in the chat.
                // 6. ENDING THE CONVERSATION: When the user indicates they are finished or asks for the results, instruct them to use the button.
                // 7. EXECUTION DIRECTIVE: Your very first response MUST NOT repeat any rules. Start DIRECTLY with your visual analysis.
                """
                image_file = Image.open(io.BytesIO(st.session_state.uploaded_image_data))
                chat_session = model.start_chat()
                response = chat_session.send_message([prompt, image_file])
                st.session_state.messages = chat_session.history
                st.session_state.analysis_stage = "conversation"
                st.rerun()

    if st.session_state.analysis_stage == "conversation":
        st.subheader("Refine Details with the AI", divider='rainbow')
        chat_session = model.start_chat(history=st.session_state.messages)

        for message in chat_session.history:
            if message.parts[0].text.strip().lower().startswith("// system"):
                continue
            if message.role in ["user", "model"]:
                with st.chat_message(message.role):
                    st.markdown(message.parts[0].text)

        if prompt := st.chat_input("Provide more details..."):
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.spinner("Thinking..."):
                response = chat_session.send_message(prompt)

                while len(response.candidates[0].content.parts) > 0 and response.candidates[0].content.parts[
                    0].function_call:
                    function_call = response.candidates[0].content.parts[0].function_call
                    tool_name = function_call.name
                    if tool_name in available_tools:
                        tool_args = dict(function_call.args)
                        function_to_call = available_tools[tool_name]
                        tool_response = function_to_call(**tool_args)

                        response = chat_session.send_message(
                            [genai.protos.Part(
                                function_response=genai.protos.FunctionResponse(
                                    name=tool_name,
                                    response={"content": tool_response},
                                )
                            )]
                        )
                    else:
                        break
                st.session_state.messages = chat_session.history
                st.rerun()

        if st.button("✅ All Details Provided, Calculate Final Estimate!"):
            with st.spinner("Finalizing..."):
                final_prompt = """
                Based on our entire conversation, your final and most important task is to act as an expert nutritionist and CALCULATE a detailed nutritional breakdown.

                - **Synthesize All Information:** Use every piece of information from our conversation (ingredients, preparation methods, quantities, and any data from web searches) to inform your calculations.
                - **Use Your Internal Knowledge:** For ingredients like "one large chicken breast," or if a web search failed, you must use your internal knowledge to estimate the nutritional values.
                - **Output Format:** You MUST ONLY respond with a single, valid JSON object. Do not include any other text. The object must have a "breakdown" key containing a list of items. Each item must have keys for "item", "calories", "protein_grams", "carbs_grams", and "fat_grams".
                - **Handle Uncertainty:** If, after using all your knowledge and tools, you are still truly unable to calculate a specific value, default that value to 0. But you must try to calculate first.

                Example: `{"breakdown": [{"item": "Pan-fried Chicken Kebabs (1 large breast)","calories": 550,"protein_grams": 75,"carbs_grams": 5,"fat_grams": 25}]}`

                Now, provide the final JSON response for the meal we discussed.
                """
                response = chat_session.send_message(final_prompt)
                st.session_state.final_analysis = response.text
                st.session_state.analysis_stage = "results"
                st.rerun()

    if st.session_state.analysis_stage == "results":
        st.success("### Here is your detailed nutritional estimate:", icon="🎉")
        raw_text = st.session_state.get("final_analysis", "")
        try:
            json_start = raw_text.find('{')
            json_end = raw_text.rfind('}') + 1
            if json_start == -1 or json_end == 0:
                raise ValueError("No valid JSON object found in the AI's response.")
            json_str = raw_text[json_start:json_end]
            data = json.loads(json_str)
            breakdown_list = data.get("breakdown", [])
            if not breakdown_list:
                st.warning("The AI was unable to provide a breakdown. Please try again.")

            total_calories, total_protein, total_carbs, total_fat = 0, 0, 0, 0
            display_data = []

            for item in breakdown_list:
                try:
                    calories = int(item.get("calories"))
                except (ValueError, TypeError):
                    calories = 0

                try:
                    protein = int(item.get("protein_grams"))
                except (ValueError, TypeError):
                    protein = 0

                try:
                    carbs = int(item.get("carbs_grams"))
                except (ValueError, TypeError):
                    carbs = 0

                try:
                    fat = int(item.get("fat_grams"))
                except (ValueError, TypeError):
                    fat = 0

                total_calories += calories
                total_protein += protein
                total_carbs += carbs
                total_fat += fat
                display_data.append(
                    {"Item": item.get("item", "N/A"), "Calories": f"{calories} kcal", "Protein": f"{protein}g",
                     "Carbs": f"{carbs}g", "Fat": f"{fat}g"})

            st.table(display_data)
            st.subheader("Calculated Totals", divider='rainbow')
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Calories", f"{total_calories} kcal")
            col2.metric("Total Protein", f"{total_protein}g")
            col3.metric("Total Carbs", f"{total_carbs}g")
            col4.metric("Total Fat", f"{total_fat}g")

        except (ValueError, json.JSONDecodeError) as e:
            st.error(f"Could not parse the final analysis. Error: {e}", icon="🤷")
            st.write("Raw AI response for debugging:")
            st.code(raw_text)

        if st.button("Start Over"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

if __name__ == "__main__":
    main()