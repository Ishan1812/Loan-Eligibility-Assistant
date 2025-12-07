import os
from typing import Literal
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

from state import LoanState
from rag_engine import query_policies

# Suppress TensorFlow and oneDNN warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Initialize LLM
# Using Ollama with Mistral
llm = ChatOllama(model="mistral", temperature=0)

# -------------------------------------
# Node Functions
# -------------------------------------

def collect_financials(state: LoanState):
    """Extract financial data from input text using LLM."""
    print("--- Node: Collect Financials ---")
    
    # Define a parser for structured output
    class Financials(BaseModel):
        salary: int = Field(description="Monthly salary")
        credit_score: int = Field(description="Credit score")
        loan_amount: int = Field(description="Loan amount requested")
        tenure_years: int = Field(description="Tenure in years")
        existing_emi: int = Field(default=0, description="Existing EMI amount")

    parser = JsonOutputParser(pydantic_object=Financials)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Extract the following financial details from the user's query. Return JSON only."),
        ("user", "{query}\n\n{format_instructions}")
    ])

    chain = prompt | llm | parser
    
    try:
        result = chain.invoke({
            "query": state.user_input,
            "format_instructions": parser.get_format_instructions()
        })
        
        # Update state
        return {
            "salary": result.get("salary"),
            "credit_score": result.get("credit_score"),
            "loan_amount": result.get("loan_amount"),
            "tenure_years": result.get("tenure_years"),
            "existing_emi": result.get("existing_emi", 0)
        }
    except Exception as e:
        print(f"Extraction failed: {e}")
        return {} # Return empty update on failure

def check_eligibility(state: LoanState):
    """Retrieve loan rules via RAG & compare with user profile."""
    print("--- Node: Check Eligibility ---")
    
    # 1. Retrieve rules
    # Construct a query based on user profile to get relevant rules
    query = f"eligibility for salary {state.salary} credit score {state.credit_score} loan {state.loan_amount}"
    rules = query_policies(query)
    
    # 2. Check eligibility using LLM
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a bank loan officer. Check eligibility based on the rules provided."),
        ("user", """
        User Profile:
        Salary: {salary}
        Credit Score: {credit_score}
        Loan Amount: {loan_amount}
        Tenure: {tenure_years}
        Existing EMI: {existing_emi}

        Bank Rules:
        {rules}

        Determine if the user is Eligible or Not Eligible. 
        Calculate the DTI (Debt to Income) ratio. DTI = (Existing EMI + New EMI) / Salary.
        Assume New EMI approx = Loan Amount / (Tenure * 12) (roughly for DTI check).
        
        Return a short summary starting with 'Eligible' or 'Not Eligible'.
        """)
    ])
    
    chain = prompt | llm
    result = chain.invoke({
        "salary": state.salary,
        "credit_score": state.credit_score,
        "loan_amount": state.loan_amount,
        "tenure_years": state.tenure_years,
        "existing_emi": state.existing_emi,
        "rules": rules
    })
    
    status = result.content
    return {"retrieved_rules": rules, "eligibility_status": status}

def predict_approval_chance(state: LoanState):
    """Estimate approval probability (simple heuristic)."""
    print("--- Node: Predict Approval Chance ---")
    
    score = 0
    # Base score
    score += 50
    
    # Credit Score impact
    if state.credit_score:
        if state.credit_score >= 800:
            score += 30
        elif state.credit_score >= 750:
            score += 20
        elif state.credit_score >= 700:
            score += 10
        else:
            score -= 20
            
    # Salary impact
    if state.salary and state.salary > 50000:
        score += 10
        
    # Cap at 99, min 0
    final_score = max(0, min(99, score))
    
    # If explicitly not eligible from previous step, lower the score
    if state.eligibility_status and "Not Eligible" in state.eligibility_status:
        final_score = min(final_score, 20)
        
    return {"approval_score": final_score}

def suggest_loan_plan(state: LoanState):
    """Calculate EMI & suggest a safe loan structure."""
    print("--- Node: Suggest Loan Plan ---")
    
    P = state.loan_amount
    N = state.tenure_years * 12
    
    # Determine interest rate based on credit score (simplified from rules)
    R_annual = 13 # default
    if state.credit_score >= 800:
        R_annual = 9
    elif state.credit_score >= 750:
        R_annual = 11
        
    R = R_annual / (12 * 100)
    
    # EMI Formula: [P x R x (1+R)^N]/[(1+R)^N-1]
    try:
        emi = (P * R * ((1 + R) ** N)) / (((1 + R) ** N) - 1)
    except:
        emi = 0
        
    # Calculate DTI
    total_emi = state.existing_emi + emi
    dti = (total_emi / state.salary) * 100 if state.salary else 0
    
    plan = f"""
    Loan Amount: ₹{P}
    Tenure: {state.tenure_years} years
    Interest Rate: {R_annual}%
    Estimated EMI: ₹{int(emi)}
    DTI After Loan: {int(dti)}%
    """
    return {"suggested_plan": plan}

def review_response(state: LoanState):
    """Format final output neatly for the user."""
    print("--- Node: Review Response ---")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Format the final response for the user."),
        ("user", """
        User Input: {user_input}
        Eligibility: {eligibility_status}
        Approval Chance: {approval_score}%
        Suggested Plan: {suggested_plan}
        
        Create a clean, professional response.
        with best regards from Ishan Garg, Loan Officer, LTIMindtree Bank of India
        """)
    ])
    
    chain = prompt | llm
    result = chain.invoke({
        "user_input": state.user_input,
        "eligibility_status": state.eligibility_status,
        "approval_score": state.approval_score,
        "suggested_plan": state.suggested_plan
    })
    
    return {"final_reply": result.content}


# -------------------------------------
# Build the Workflow Graph
# -------------------------------------

workflow = StateGraph(LoanState)

workflow.add_node("collect_financials", collect_financials)
workflow.add_node("check_eligibility", check_eligibility)
workflow.add_node("predict_approval_chance", predict_approval_chance)
workflow.add_node("suggest_loan_plan", suggest_loan_plan)
workflow.add_node("review_response", review_response)

workflow.add_edge("collect_financials", "check_eligibility")
workflow.add_edge("check_eligibility", "predict_approval_chance")
workflow.add_edge("predict_approval_chance", "suggest_loan_plan")
workflow.add_edge("suggest_loan_plan", "review_response")
workflow.add_edge("review_response", END)

workflow.set_entry_point("collect_financials")
loan_app = workflow.compile()

if __name__ == "__main__":
    # Example Run
    user_query = "I earn 105k/month, credit score 740, loan required 1 lakhs for 3 yrs. Already paying emi 12k."
    print(f"Processing query: {user_query}\n")
    
    try:
        result = loan_app.invoke({"user_input": user_query})
        print("\n=== FINAL RESPONSE ===\n")
        print(result["final_reply"])
    except Exception as e:
        print(f"Error during execution: {e}")
        print("Ensure Ollama is running and 'mistral' model is available.")
