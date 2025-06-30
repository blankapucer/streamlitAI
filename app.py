# Fix SQLite version issue for ChromaDB on Streamlit Cloud
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass
    
# Simple Q&A App using Streamlit
# Students: Replace the documents below with your own!

# IMPORTS - These are the libraries we need
import streamlit as st          # Creates web interface components
import chromadb                # Stores and searches through documents  
from transformers import pipeline  # AI model for generating answers

def setup_documents():
    """
    This function creates our document database
    NOTE: This runs every time someone uses the app
    In a real app, you'd want to save this data permanently
    """
    client = chromadb.Client()
    try:
        collection = client.get_collection(name="docs")
    except Exception:
        collection = client.create_collection(name="docs")
    
    
    # STUDENT TASK: Replace these 5 documents with your own!
    # Pick ONE topic: movies, sports, cooking, travel, technology
    # Each document should be 150-200 words
    # IMPORTANT: The quality of your documents affects answer quality!
    
    my_documents = [
        """Nutrition is the process by which your body takes in and uses food to maintain health, grow, and repair itself. It involves macronutrients—carbohydrates, proteins, and fats—that provide energy and building blocks, and micronutrients—vitamins and minerals—that support vital functions.
Good nutrition is essential because it influences every system in your body. Without it, you can face problems like fatigue, poor immunity, or chronic diseases.
Globally, nutrition presents a paradox: while about 820 million people suffer from hunger and malnutrition, over 2 billion adults are overweight or obese (WHO, 2021). This reflects not just a lack of food, but poor-quality diets.
For example, in many countries, people consume enough calories but not enough nutrients, leading to “hidden hunger” where vitamin and mineral deficiencies occur despite eating enough.
Balanced nutrition means eating a variety of foods—whole grains, lean proteins, fruits, and vegetables—to meet your body’s needs. Understanding nutrition basics sets the foundation for better health and wellbeing.
""",
        
        """Macronutrients—carbohydrates, proteins, and fats—are the body’s primary energy sources.
Carbohydrates supply about 4 calories per gram and should make up 45–65% of your daily calories (Dietary Guidelines for Americans, 2020). Complex carbs, such as whole grains, beans, and vegetables, release energy steadily and provide fiber, which aids digestion. For example, an average adult eating 2,000 calories should consume about 225–325 grams of carbs daily.
Proteins also provide 4 calories per gram and are vital for building muscles, enzymes, and tissues. The Recommended Dietary Allowance (RDA) for protein is 0.8 grams per kilogram of body weight. For a 70 kg (154 lbs) person, that equals about 56 grams daily. Athletes may require more—up to 1.2–2.0 grams/kg.
Fats are the most energy-dense macronutrient, offering 9 calories per gram. Healthy fats should comprise 20–35% of daily calories. For a 2,000-calorie diet, this is roughly 44–78 grams of fat daily. Sources include olive oil, nuts, seeds, and fatty fish rich in omega-3s.
Imbalances—like consuming over 60% of calories from unhealthy fats or very low protein intake—can increase risks of heart disease or muscle loss. A balanced approach with quality sources supports long-term health.

""",
        
        """Micronutrients are vitamins and minerals needed in small amounts but essential for health.
Vitamin A supports vision and immunity, with a recommended daily intake of 900 mcg for men and 700 mcg for women (NIH). It’s found in sweet potatoes, carrots, and leafy greens.
Vitamin C acts as an antioxidant and aids healing. Adults need about 90 mg daily for men and 75 mg for women. One medium orange provides around 70 mg.
Vitamin D regulates calcium and bone health; the RDA is 600 IU (15 mcg) for most adults, increasing to 800 IU after age 70. Sun exposure helps, but fortified dairy and fatty fish are good dietary sources.
Iron is crucial for oxygen transport, with men needing 8 mg/day and women of reproductive age requiring 18 mg/day due to menstruation. Deficiency affects over 1.6 billion people worldwide (WHO), causing anemia and fatigue.
Calcium supports bones; adults need 1,000 mg daily, increasing to 1,200 mg for women over 50. Dairy products and fortified plant milks are excellent sources.
Zinc supports immunity; the RDA is 11 mg for men and 8 mg for women. Deficiency can impair immune function.
Eating a varied diet rich in colorful fruits, vegetables, whole grains, and lean proteins usually meets these needs. In populations with limited access to diverse foods, supplementation programs help prevent micronutrient deficiencies.
""",
        
        """Water is often overlooked but is essential for nutrition and overall health. It makes up about 60% of the human body and is critical for digestion, nutrient transport, temperature regulation, and joint lubrication.
The average recommendation is about 2.7 liters (91 ounces) per day for women and 3.7 liters (125 ounces) for men, including fluids from food and drinks (National Academies, 2020).
Dehydration can cause fatigue, headaches, poor concentration, and, in severe cases, kidney damage. Even mild dehydration reduces physical and mental performance.
Besides plain water, many fruits and vegetables—like watermelon, cucumbers, and oranges—are excellent hydration sources because they contain up to 90% water. Herbal teas and milk also contribute.
To stay hydrated, carry a water bottle, drink regularly throughout the day, and adjust intake based on activity, weather, and health status. Proper hydration supports every aspect of nutrition by helping your body absorb and use nutrients efficiently.
""",
        
        """Nutrition plays a powerful role in preventing chronic diseases like heart disease, type 2 diabetes, and certain cancers. These conditions are linked to poor diet and lifestyle habits.
For example, a diet high in saturated fats and sugar increases heart disease risk, while fiber-rich diets lower it. The American Heart Association recommends consuming at least 25–30 grams of fiber daily from fruits, vegetables, and whole grains.
Antioxidants—found in berries, nuts, and dark leafy greens—help protect cells from damage that can lead to cancer and aging.
Research, such as the 2015 Dietary Guidelines for Americans, stresses plant-based diets rich in whole foods for disease prevention. Studies show Mediterranean and DASH diets reduce heart disease risk by up to 30%.
Practical tips include reducing processed foods, limiting added sugars, and eating more colorful vegetables and lean proteins. Nutrition is not just about weight but also about building a body resilient to long-term illnesses.
"""
    ]
    
    # Add documents to database with unique IDs
    # ChromaDB needs unique identifiers for each document
    collection.add(
        documents=my_documents,
        ids=["doc1", "doc2", "doc3", "doc4", "doc5"]
    )
    
    return collection

def get_answer(collection, question):
    """
    This function searches documents and generates answers while minimizing hallucination
    """
    
    # STEP 1: Search for relevant documents in the database
    # We get 3 documents instead of 2 for better context coverage
    results = collection.query(
        query_texts=[question],    # The user's question
        n_results=3               # Get 3 most similar documents
    )
    
    # STEP 2: Extract search results
    # docs = the actual document text content
    # distances = how similar each document is to the question (lower = more similar)
    docs = results["documents"][0]
    distances = results["distances"][0]
    
    # STEP 3: Check if documents are actually relevant to the question
    # If no documents found OR all documents are too different from question
    # Return early to avoid hallucination
    if not docs or min(distances) > 1.5:  # 1.5 is similarity threshold - adjust as needed
        return "I don't have information about that topic in my documents."
    
    # STEP 4: Create structured context for the AI model
    # Format each document clearly with labels
    # This helps the AI understand document boundaries
    context = "\n\n".join([f"Document {i+1}: {doc}" for i, doc in enumerate(docs)])
    
    # STEP 5: Build improved prompt to reduce hallucination
    # Key changes from original:
    # - Separate context from instructions
    # - More explicit instructions about staying within context
    # - Clear format structure
    prompt = f"""Context information:
{context}

Question: {question}

Instructions: Answer ONLY using the information provided above. If the answer is not in the context, respond with "I don't know." Do not add information from outside the context.

Answer:"""
    
    # STEP 6: Generate answer with anti-hallucination parameters
    ai_model = pipeline("text2text-generation", model="google/flan-t5-small")
    response = ai_model(
        prompt, 
        max_length=150
    )
    
    # STEP 7: Extract and clean the generated answer
    answer = response[0]['generated_text'].strip()
    

    
    # STEP 8: Return the final answer
    return answer

# MAIN APP STARTS HERE - This is where we build the user interface

# STREAMLIT BUILDING BLOCK 1: PAGE TITLE
# st.title() creates a large heading at the top of your web page
# The emoji 🤖 makes it more visually appealing
# This appears as the biggest text on your page
st.title("🍵🥛🍃Nutrition 101🍃🥛🍵")

# STREAMLIT BUILDING BLOCK 2: DESCRIPTIVE TEXT  
# st.write() displays regular text on the page
# Use this for instructions, descriptions, or any text content
# It automatically formats the text nicely
st.write("Welcome to the Nutrition 101 database! Ask me anything about nutrition.")

# STREAMLIT BUILDING BLOCK 3: FUNCTION CALLS
# We call our function to set up the document database
# This happens every time someone uses the app
collection = setup_documents()

# STREAMLIT BUILDING BLOCK 4: TEXT INPUT BOX
# st.text_input() creates a box where users can type
# - First parameter: Label that appears above the box
# - The text users type gets stored in the 'question' variable
# - Users can click in this box and type their question
question = st.text_input("Do you have any burning questions about nutririon?")

# STREAMLIT BUILDING BLOCK 5: BUTTON
# st.button() creates a clickable button
# - When clicked, all code inside the 'if' block runs
# - type="primary" makes the button blue and prominent
# - The button text appears on the button itself
if st.button("Find the right answer!", type="primary"):
    
    # STREAMLIT BUILDING BLOCK 6: CONDITIONAL LOGIC
    # Check if user actually typed something (not empty)
    if question:
        
        # STREAMLIT BUILDING BLOCK 7: SPINNER (LOADING ANIMATION)
        # st.spinner() shows a rotating animation while code runs
        # - Text inside quotes appears next to the spinner
        # - Everything inside the 'with' block runs while spinner shows
        # - Spinner disappears when the code finishes
        with st.spinner("Getting answer..."):
            answer = get_answer(collection, question)
        
        # STREAMLIT BUILDING BLOCK 8: FORMATTED TEXT OUTPUT
        # st.write() can display different types of content
        # - **text** makes text bold (markdown formatting)
        # - First st.write() shows "Answer:" in bold
        # - Second st.write() shows the actual answer
        st.write("**Answer:**")
        st.write(answer)
    
    else:
        # STREAMLIT BUILDING BLOCK 9: SIMPLE MESSAGE
        # This runs if user didn't type a question
        # Reminds them to enter something before clicking
        st.write("Please enter a question!")

# STREAMLIT BUILDING BLOCK 10: EXPANDABLE SECTION
# st.expander() creates a collapsible section
# - Users can click to show/hide the content inside
# - Great for help text, instructions, or extra information
# - Keeps the main interface clean
with st.expander("About 🍵🥛🍃Nutrition 101🍵🥛🍃"):
    st.write("""
   I created this app to help you learn about nutrition. The main topics include:
            What is nutrition and why it matters;
            Macronutrients: carbs, proteins, and fats;
            Micronutrients: vitamins and minerals;
            The role of hydration;
            Nutririon and chronic disease prevention.

    
    Ask me any question about these topics, and I will do my best to provide a helpful answer based on the information in my database.
    """)

# Add colored text using markdown
st.markdown("### 🍎🥗 Welcome to **Nutrition 101**!")
st.markdown("*Your personal assistant to improve your well-being one meal at a time*")

# Personalize the spinner message
with st.spinner("Searching my nutrition database..."):
    answer = get_answer(collection, question)

    # Custom success/error messages
    st.success("🥦 Found the perfect answer for you!")
    st.info("🥑Tip: Try asking about specific nutrition topics!")
    
 # Add a pastel background using custom CSS
    st.markdown(
        """
        <style>
        body {
            background: linear-gradient(135deg, #f8fafc 0%, #e0f7fa 40%, #ffe0f0 100%);
        }
        .stApp {
            background: linear-gradient(135deg, #f8fafc 0%, #e0f7fa 40%, #ffe0f0 100%);
        }
        </style>
        """,
        unsafe_allow_html=True)

# TO RUN: Save as app.py, then type: streamlit run app.py

