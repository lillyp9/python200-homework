import json
from dotenv import load_dotenv
from openai import OpenAI

#load the .env file where API key is stored 
load_dotenv()
client = OpenAI()
#start OpenAI client
def get_completion(messages, model="gpt-4o-mini", temperature=0.7):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=400
    )
    return response.choices[0].message.content

#System Prompt set up the model as A JOB APPLICATIION COACH.

system_prompt = """ You are a job application coach helping the individual swithc to a career change where tranistioning their past experience into a way that resonates with the hiring managers in the new field.
Your individual is a typically profesionals shifting industries. For example, ateacher moving to coperate training, or a nurse moving into a health-tech product roles.
Your job as a job application coach is to help them:
- rewrite resume bullet points so the trasnferable skills are clearly higlighted.
- Draft a cover-letter paragraphs that connect their past experience to the new role.
- Answer follwo-up questions about tone, how to phrase and to talk about a career change.
Rules you have to follow based on your behavior:
1. Stay focused on the job applicatiion materials. Do not stray away from it . If your individual ask something else that is off-topic (non-job related , personal life), poitely decine and stteer them nack to the resume and cover letter.
2. At the end of EVERY response including drafted content ( bullet, paragraphs, or letter), remind the individual to review and edit the output before sending it anywhere. AI-generated application material should always be checked by a human.
3. Keep in mind you do not know the individual's specific industry norms,  target company or its personal voice/opinions. If there is doubt, say so and ask the individual to clarify or use their own judgement rather then guessing.
4. Keep the tone of your reponses focused and practical.  No overly cheerfullness. Talk to them like a coach who is working through the job application process with them. """

# System Prompt design: I created the rule 2 to remind the individual to review any of the response/output the modle gives out.
#It is not a one time reminder , given that the model can generate multiple responses, I want the model to remind them to check the output everytime.
#AI-generated content sometimes gives hallunications or may noot be accurate, so it is very important for a human to review. It baiscally a safety precaution that shold matter in the model. Sedning a unedited AI cover letter to a real recruiter would be a disaster , so this is crucial.

print("---System Prompt--")
print(system_prompt)

#========== Task 2: Bullet Point Rewriter =======
def rewrite_bullets(bullets: list[str]) -> list[dict]:
    # Format the bullets into a delimited block
    bullet_text = "\n".join(f"- {b}" for b in bullets)

    prompt = f"""
    You are a professional resume coach helping a career changer.
    Rewrite each resume bullet point below to be more specific, results-oriented, and compelling.
    Use strong action verbs. Do not invent facts that aren't implied by the original.
    Where the original is vague , include a placeholder like "[X%]" or "[X amount]" to indicate where the individual should add real numbers or details.
    For each bullet point, include a confidence score from 0.0 to 1.0, showing how confident you are in rewrite.  Lower the score when the original is too vague.
    Respond with raw JSON only. DO NOT wrap the ouput in markdown block. Do not add any text before or after the JSON.The first character of your response must
be "[" and the last must be "]".The JSON should be a list where each item has three keys: "original", "improved", and "confidence".

Bullet points:
{bullet_text}
"""
    messages = [{"role": "user", "content": prompt}]
    response_text = get_completion(messages, temperature=0.5)
    #Clean up the markdown code if the model adds them.
    cleaned = response_text.strip()
    if cleaned.startswith("'''"):
        #Drop first line ''' json and last line ''' json
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned
        cleaned = cleaned.rsplit("```", 1)[0]
        cleaned = cleaned.strip()
    try:
        results = json.loads(cleaned)
    except json.JSONDecodeError:
        print("Failed to parse JSON. Raw response:")
        print(response_text)
        raise
  

    #print side by side comparison
    print("---Bullet Point Rewriter---")
    for i, item in enumerate(results, start=1):
        print(f"Bullet {i}")
        print(f"Original: {item['original']}")
        print(f"Improved: {item['improved']}")
        print(f"Confidenece: {item['confidence']:.2f}")
        #the confidence score 
        if item['confidence'] < 0.7:
            print("Confidence is low , rewrite this bullet point again with more specific details.")
        print()
    return results        
  #test the function with examples.
bullets = [
    "Help customers with their issues.",
    "Made reports and presentation for the team.",
    "Work with other departments to complete projects."
]
results = rewrite_bullets(bullets)

#Before ading the clean up code , the model was adding markdowns and it was causing the json parsing to fail.
#By adding the clean up code, it checks if the response starts with ''' and if it does, it removes the first line and the last line of the response. This allows us to get the raw JSON without any markdowns, which can then be parsed successfully.

#These starter bullets are very weak and vague, such as "Help customers with their issues." It does not tell the recruiter nothing  about the kind of issues, how many customers were satisfyed, or what difference did the help make.
#After the rewrite and the improvement, the bullet point changed to "Resolved customer inquiries and issues, enhancing satisfaction levels by [X%] through effective communication and problem-solving."
#The improved bulet was more specific and  including the placeholder so the individual can add the details or numbers.
#Comapring both original and imrpoved bullet , the improved sounds more like a accomplishments.


#================= Task 3 Cover -Letter Generator ========
def generate_cover_letter(job_title: str, background: str) -> str:
    prompt = f"""
    You write strong cover letter opening paragraphs for career changers.
    The paragraph should be 3-5 sentences: confident, specific, and free of clichés.

    Here are two examples of the style and tone you should match:

    Example 1:
    Role: Data Analyst at a healthcare nonprofit
    Background: Seven years as a registered nurse, recently completed a data analytics bootcamp.
    Opening: After seven years as a registered nurse, I've spent my career making decisions
    under pressure using incomplete information — which turns out to be excellent training for
    data analysis. I recently completed a data analytics program where I built dashboards
    tracking patient outcomes across departments. I'm excited to bring that combination of
    clinical context and technical skill to [Company]'s mission-driven work.

    Example 2:
    Role: Junior Software Engineer at a fintech startup
    Background: Ten years in retail banking operations, self-taught Python developer for two years.
    Opening: I spent a decade on the operations side of banking, watching technology decisions
    get made by people who had never processed a wire transfer or resolved a failed ACH batch.
    That frustration turned into curiosity, and two years of self-teaching Python later, I'm
    ready to be on the other side of those decisions. I'm applying to [Company] because your
    work on payment infrastructure is exactly where my domain expertise and new technical skills
    intersect.

    Now write an opening paragraph for this person:
    Role: {job_title}
    Background: {background}
    Opening:
    """

    messages = [{"role": "user", "content": prompt}]
    paragraph = get_completion(messages, temperature=0.7)
    return paragraph 

#test function with example
job_title = "Product Manager at a health-tech company"
background = "Eight years as a physical therapist, recently completed a product management certification program."
print("\n---Cover Letter Opening ---")
cover_letter = generate_cover_letter(job_title, background)
print(cover_letter)

#I choosed the two example (pphysical therapist to product manager) or (nurse to data analyist) because it shows the model how to connect past experinece to new role.
#Drawing a connectiion to the new role and avoide general overuse phrases like "Passionate professional" or "unique skills" is important to make the cover letter sound more specific and less cliche.
#The model used placeholder in example for the company name whiich trains the model to not make up a companys name.
#The few-shot prompting with the examples help the model control the style, tone , structure and more clairty with the output.
#the zero-shot could work as well but the output would be more generic ai-generated.
#Showing the model two examples that share the same pattern and structure  made it easier for the model to understand  what task to do.

#============ Task 4 Moderation Check ===========
def is_safe(text: str) -> bool:
    result = client.moderations.create(
        model="omni-moderation-latest",
        input=text
    )
    flagged = result.results[0].flagged

    #conditional statement 
    if  flagged:
        print("Your text has been flaggged by the moderation check system. " \
        "Please revise your text and try again.")
        return False
    return True

    #test the function out 
print("\n---Moderation Check---")
    #Safe input that should pass 
safe_input = "I am excited to apply for this role , Can you help me write my resume for this marketing position?"
print(f"Test 1 - Safe Input: {safe_input!r}")
print(f"Result: {is_safe(safe_input)}")

#Input that should be flagged 
unsafe_input = " I hate this company and want to destory their company even if it means ruining their reputation."
print(f"\nTest 2 - Unsafe Input: {unsafe_input!r}")
print(f"Result: {is_safe(unsafe_input)}")

#Test the borderline 
borderline_input = "The job interview process is furstrating, I feel dead."
print(f"Test 3(borderline): {borderline_input!r} ")

#Get the full moderation result for the borderline input 
result = client.moderations.create(
    model = "omni-moderation-latest",
    input = borderline_input
)
print(f"Flagged: {result.results[0].flagged}")
print("Category breakdown:")
for category, value in result.results[0].categories.model_dump().items():
    print(f" {category}: {value}")

print("\n Category scores (how confident the model is):")
for category, score in result.results[0].category_scores.model_dump().items():
    print(f" {category}: {score:.4f}")

#The borderline proves the filter works its not just randomly flagged. Such as a phrase I feel dead , containing the word "dead".
#It doesnt flag it since the model understand the context. 
#The moderation check also has catergories with scores that show how confident the model is that the text falls into that category.
#This is useful for understanding why the text is flagged and help the individual revise it or watch what they type.



#========= Task 5: The Chatbot Loop =====
def run_chatbot():
    # 1. Initialize conversation history with your system prompt
    messages = [
        {"role": "system", "content": system_prompt}
    ]

    print("=" * 50)
    print("Job Application Helper")
    print("=" * 50)
    print("I can help you with:")
    print("  1. Rewriting resume bullet points")
    print("  2. Drafting a cover letter opening")
    print("  3. Any other questions about your application")
    print("\nType 'quit' at any time to exit.\n")

    while True:
        user_input = input("You: ").strip()

        # 2. Handle exit
        if user_input.lower() in {"quit", "exit"}:
            print("\nJob Application Helper: Good luck with your applications!")
            break

        # 3. Skip empty input
        if not user_input:
            continue

        # 4. Run moderation check before doing anything else
        if not is_safe(user_input):
            continue  # is_safe() already printed the warning message

            # 5. Check if the user wants to rewrite bullets
        #    (hint: look for keywords like "bullet" or "resume" in user_input.lower())
        if "bullet" in user_input.lower() or "resume" in user_input.lower():
            print("\nJob Application Helper: Paste your bullet points below, one per line.")
            print("When you're done, type 'DONE' on its own line.\n")
            raw_bullets = []
            while True:
                line = input().strip()
                if line.upper() == "DONE":
                    break
                if line:
                    raw_bullets.append(line) 

            #Call the function rewrite_bullets() and print results
            if raw_bullets:
                rewrite_bullets(raw_bullets)
            else:
                print("Job Application Helper: No bullets to rewrite. Please try again.")
          
          # 6. Check if the user wants a cover letter
        elif "cover letter" in user_input.lower():
            job_title = input("Job Application Helper: What is the job title? ").strip()
            background = input("Job Application Helper: Briefly describe your background: ").strip()   

            if job_title and background:
                letter = generate_cover_letter(job_title, background)
                print(f"\nJob Application Helper:\n{letter}\n")
                print("Remember to review and edit this before sending.")
            else:
                print("Job Application Helper: I need both the job title and background. Try again please.")   

                # 7. Otherwise, handle it as a regular chat turn
        else:
            #Add message history
            messages.append({"role": "user", "content": user_input})

            #get the model response
            response = get_completion(messages)

            #show it to individual
            print(f"Job Application Helper:\n{response}\n")

            #Add the response to message history
            messages.append({"role": "assistant", "content": response})
            
            #print(f"[DEBUG: messages list length = {len(messages)}]")
            pass

#Run the chatbot
if __name__ == "__main__":
    run_chatbot()


#======= Questions =======
#What could go wrong if a job-seeker submitted the bot's output directly — without reviewing it — to a real employer?
#If a job-seeker submits the bots output directly without reviewing it, there would be sveral issues.
# The bot could fabricate details and credentials that the indivdual does not have. The bot could invent and accomplishment , make up numbers/percentages. 
# While at first it can look impressive, it would be a disaster if th recruiter is interviewingthem  and all of a sudden say "Tell me more of X and Y" and the individual has no clue in how to answer.
# But it is in their resume. Not to mention recruiters are trained into noticing right away an AI-generated text and if the tone of voice doeosnt match what its written in the resume its a red flag autaomatic.

#What is one guardrail you would add if you were deploying this tool professionally? (A guardrail is any design choice that reduces the chance of harm — a UI warning, a moderation filter, a usage policy, a disclaimer, or something else entirely.)
#Adding a disclaimer at the start of every session that AI can hallucinate, double check if the information is right. By doing this you are letting the individual know not to fully trust what the AI output.

