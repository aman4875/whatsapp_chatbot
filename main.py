from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from sentence_transformers import SentenceTransformer, util
import openai
import os
from dotenv import load_dotenv
import re

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)
user_states = {}

# FAQs and Answers
FAQ_LIST = [
    # Services & Expertise
    ("What services do you offer?", 
     "We offer a full range of IT and digital solutions, including web & mobile app development, UI/UX design, custom software, SaaS, e-commerce, and AI/ML solutions."),
    
    ("Can you develop custom web or mobile applications?", 
     "Yes, we build custom web and mobile applications tailored to your needs."),
    
    ("Do you provide UI/UX design?", 
     "Absolutely! We have a dedicated UI/UX team to design intuitive digital experiences."),
    
    ("Are e-commerce solutions available?", 
     "Yes, we deliver powerful e-commerce solutions with custom features, carts, payments, etc."),
    
    ("What technologies do you specialize in?", 
     "We work with modern stacks like React, Node, Django, Flutter, Laravel, and more."),
    
    ("Can you work with specific platforms like Shopify, WooCommerce, WordPress?", 
     "Yes! We can build using Shopify, WooCommerce, WordPress, and custom CMS."),
    
    # Project Inquiries & Quoting
    ("How do I get a quote for my project?", 
     "Please share your project details, and we’ll provide a tailored quote with budget estimates and a proposed timeline."),
    
    ("What is your process for starting a project?", 
     "We begin with understanding your goals, propose a technical solution, share budget and timeline estimates, and start the design and development after your approval."),
    
    ("How long will my project take?", 
     "Timeline depends on the scope. Share your requirements for a personalized estimate, or book a meeting with our experts."),
    
    ("What is the estimated budget for my requirements?", 
     "Cost depends on scope and complexity. Send us your requirements for a custom quote."),
    
    ("How can I book a call or meeting with your team?", 
     "You can schedule a meeting here 👉 https://bytecodetechnologies.in/book-demo"),
    
    # Company Information
    ("What is Bytecode Technologies' background?", 
     "Bytecode Technologies is a digital product agency with 300+ successful apps across industries."),
    
    ("What industries do you have experience with?", 
     "Our team has delivered over 300 applications in FinTech, Healthcare, EduTech, Real Estate, and more."),
    
    ("How many projects have you completed?", 
     "We've successfully delivered over 300 projects globally."),
    
    ("Why should I choose Bytecode Technologies?", 
     "Bytecode Technologies merges strong technical expertise with client-centered service, offering innovative, scalable solutions to power business growth."),
    
    # Support & After-Sales
    ("What kind of after-sales support do you provide?", 
     "Yes, our support team ensures robust after-sales service, helping with maintenance, upgrades, and issue resolution."),
    
    ("How do I get help with an ongoing project?", 
     "If you're an existing client, just message here or email your PM. We’re always available."),
    
    ("Who do I contact for technical issues?", 
     "Please describe the issue and we'll connect you with a tech specialist."),
    
    # Other
    ("Are you hiring?", 
     "We’re always looking for talented professionals. Visit our Careers page or ask for current opportunities."),
    
    ("How do I contact your team?", 
     "You can type your query here, email us, or request a call/meeting with our specialists for a detailed discussion.")
]

# Load embedding model once
model = SentenceTransformer('all-MiniLM-L6-v2')
faq_questions = [q for q, _ in FAQ_LIST]
faq_answers = [a for _, a in FAQ_LIST]
faq_embeddings = model.encode(faq_questions, normalize_embeddings=True)

# Main menu
MAIN_MENU = """👋 Hello! Welcome to *Bytecode Technologies* — Your Digital Partner.

How can we help you code your next success?

1️⃣ Know our services  
2️⃣ Request a quote  
3️⃣ Book a call  
4️⃣ Ask a custom question  
5️⃣ Support for ongoing project

Type a number or just ask your question!
"""

def normalize_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text.strip()

# --- GPT fallback ---
def get_gpt_response(user_input):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an assistant for Bytecode Technologies.\n\n"
                        "Only answer questions related to:\n"
                        "- Bytecode Technologies (services, pricing, hiring, support)\n"
                        "- Our technologies, tools, frameworks, infrastructure\n"
                        "- Our projects, work process, clients, payment systems, deployment, timelines, etc.\n\n"
                        "If the user asks something unrelated to our company (e.g. 'What is gravity?', 'Who is the PM?', or jokes), reply with:\n"
                        "\"I'm here to help with Bytecode Technologies-related queries 😊.\"\n\n"
                        "Be flexible and smart: if a user asks follow-up questions about something we've already discussed (e.g. tech stack or pricing), you can respond naturally."
                    )
                },
                {"role": "user", "content": user_input}
            ],
            temperature=0.5
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        print("GPT error:", str(e))
        return "Sorry, I'm having trouble answering right now."

# --- Fuzzy match with fallback ---
def find_best_faq_answer(user_input, threshold=0.7):
    query_embedding = model.encode(normalize_text(user_input), normalize_embeddings=True)
    scores = util.cos_sim(query_embedding, faq_embeddings)[0]
    best_idx = scores.argmax().item()
    best_score = scores[best_idx].item()

    if best_score >= threshold:
        return faq_answers[best_idx]
    else:
        with open("unanswered_questions.txt", "a") as f:
            f.write(user_input + "\n")
        return get_gpt_response(user_input)

# --- Main WhatsApp Route ---
@app.route("/", methods=["POST"])
def whatsapp():
    sender = request.form.get("From")
    msg = request.form.get("Body", "").strip().lower()
    resp = MessagingResponse()
    reply = resp.message()

    if msg in ["restart", "menu"]:
        user_states[sender] = {"step": 0}
        reply.body(MAIN_MENU)
        return str(resp)

    if sender not in user_states:
        user_states[sender] = {"step": 0}
        reply.body(MAIN_MENU)
        return str(resp)

    state = user_states[sender]

    # Step 0: Menu
    if state["step"] == 0:
        if msg in ["1", "services"]:
            reply.body(faq_answers[0])
            state["last_topic"] = "services"
            return str(resp)
        elif msg in ["2", "quote"]:
            state["step"] = 1
            reply.body("Great! What's your name?")
            return str(resp)
        elif msg in ["3", "meeting"]:
            reply.body("You can book a call 👉 https://bytecodetechnologies.in/book-demo")
            return str(resp)
        elif msg in ["5", "support"]:
            reply.body("Sure! Please describe your issue and we’ll connect you with a tech specialist.")
            return str(resp)
        else:
            # 👇 Check for follow-up after "services"
            if state.get("last_topic") == "services":
                service_faqs = FAQ_LIST[0:6]
                questions = [q for q, _ in service_faqs]
                answers = [a for _, a in service_faqs]
                embeddings = model.encode(questions)
                query_embedding = model.encode(msg)
                scores = util.cos_sim(query_embedding, embeddings)[0]
                best_idx = scores.argmax().item()

                if scores[best_idx] >= 0.6:
                    reply.body(answers[best_idx])
                    return str(resp)

            answer = find_best_faq_answer(msg)
            reply.body(answer)
            state["step"] = 0
            return str(resp)

    # Quote Flow
    if state["step"] == 1:
        state["name"] = msg
        reply.body(f"Thanks {msg.title()}! What's your email?")
        state["step"] = 2
        return str(resp)

    if state["step"] == 2:
        state["email"] = msg
        reply.body("Got it! Can you describe your project briefly?")
        state["step"] = 3
        return str(resp)

    if state["step"] == 3:
        state["project_details"] = msg
        with open("leads.txt", "a") as f:
            f.write(f"{state['name']},{state['email']},{msg}\n")
        reply.body(f"Thanks {state['name'].title()}! We'll contact you soon. Want to book a call too? (yes/no)")
        state["step"] = 4
        return str(resp)

    if state["step"] == 4:
        if "yes" in msg:
            reply.body("Awesome! Schedule a call 👉 https://bytecodetechnologies.in/book-demo")
        else:
            reply.body("No problem. Type 'menu' to explore more options.")
        state["step"] = 0
        return str(resp)

    # Default fallback
    reply.body("I didn’t catch that. Type 'menu' to see options or rephrase your question.")
    return str(resp)

if __name__ == "__main__":
    app.run(port=5000, debug=True)