
import os
from dotenv import load_dotenv
from typing import List, Dict, Optional, Any
import json
from datetime import datetime
import uuid

# LangChain imports
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import BaseMemory
from langchain_core.callbacks import BaseCallbackHandler

# Load environment variables
load_dotenv()

class TroubleshootingCallbackHandler(BaseCallbackHandler):
    """Custom callback handler to track troubleshooting progress"""
    
    def __init__(self):
        self.current_step = None
        self.completed_steps = []
        self.user_responses = {}
    
    def on_llm_start(self, serialized, prompts, **kwargs):
        print("🔧 Analyzing issue...")
    
    def on_llm_end(self, response, **kwargs):
        print("✅ Analysis complete")

class TroubleshootingMemory(BaseMemory):
    """Custom memory for troubleshooting conversations"""
    
    def __init__(self, return_messages: bool = True):
        self.chat_memory: List = []
        self.return_messages = return_messages
        self.current_procedure = None
        self.procedure_state = {}
        self.customer_context = {}
    
    @property
    def memory_variables(self) -> List[str]:
        return ["chat_history", "procedure_state", "customer_context"]
    
    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "chat_history": self.chat_memory,
            "procedure_state": self.procedure_state,
            "customer_context": self.customer_context
        }
    
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        input_str = inputs.get("input", "")
        output_str = outputs.get("output", "")
        
        self.chat_memory.append(HumanMessage(content=input_str))
        self.chat_memory.append(AIMessage(content=output_str))
        
        # Keep only last 10 exchanges
        if len(self.chat_memory) > 20:
            self.chat_memory = self.chat_memory[-20:]
    
    def clear(self) -> None:
        self.chat_memory.clear()
        self.procedure_state.clear()
        self.customer_context.clear()

class StoreTroubleshootingAgent:
    """LangChain-based troubleshooting agent for store support"""
    
    def __init__(self, persist_directory: str = "./langchain_chroma_db"):
        """Initialize the troubleshooting agent"""
        
        # Verify API key
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("Missing OPENAI_API_KEY environment variable")
        
        # Initialize components
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.llm = ChatOpenAI(model="gpt-4", temperature=0.1)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Initialize vector store
        self.vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings,
            collection_name="troubleshooting_procedures"
        )
        
        # Initialize memory and callback
        self.memory = TroubleshootingMemory()
        self.callback_handler = TroubleshootingCallbackHandler()
        
        # Session management
        self.session_id = str(uuid.uuid4())
        self.customer_tickets = {}
        
        print("✅ Store Troubleshooting Agent initialized")
        print(f"📋 Session ID: {self.session_id}")
    
    def add_troubleshooting_procedure(self, 
                                    procedure_text: str, 
                                    procedure_name: str,
                                    category: str = "general",
                                    product_type: str = "electronics") -> None:
        """Add a troubleshooting procedure to the knowledge base"""
        
        # Parse procedure into structured format
        procedure_data = self._parse_procedure_advanced(procedure_text)
        
        # Create documents for different sections
        documents = []
        
        # Main procedure document
        main_doc = Document(
            page_content=procedure_text,
            metadata={
                "type": "complete_procedure",
                "name": procedure_name,
                "category": category,
                "product_type": product_type,
                "timestamp": datetime.now().isoformat()
            }
        )
        documents.append(main_doc)
        
        # Individual components as separate documents
        for component_type, content in procedure_data.items():
            if content:
                if isinstance(content, list):
                    content_str = "\n".join(content) if content else ""
                elif isinstance(content, dict):
                    content_str = json.dumps(content, indent=2)
                else:
                    content_str = str(content)
                
                if content_str.strip():
                    comp_doc = Document(
                        page_content=content_str,
                        metadata={
                            "type": component_type,
                            "procedure": procedure_name,
                            "category": category,
                            "product_type": product_type,
                            "parent_procedure": procedure_name
                        }
                    )
                    documents.append(comp_doc)
        
        # Split documents if needed
        split_docs = []
        for doc in documents:
            if len(doc.page_content) > 1000:
                chunks = self.text_splitter.split_documents([doc])
                split_docs.extend(chunks)
            else:
                split_docs.append(doc)
        
        # Add to vector store
        self.vectorstore.add_documents(split_docs)
        
        print(f"✅ Added procedure '{procedure_name}' with {len(split_docs)} document chunks")
    
    def _parse_procedure_advanced(self, text: str) -> Dict:
        """Advanced parsing of troubleshooting procedures"""
        import re
        
        procedure = {}
        
        # Extract sections using regex patterns
        sections = {
            'symptom': r'Symptom:\s*([^\n]+)',
            'context_questions': r'Context_Questions:\s*((?:- [^\n]+\n?)*)',
            'diagnostic_flow': r'Diagnostic_Flow:\s*((?:\d+\..*?(?=\d+\.|Success_Indicators:|Common_Resolutions:|$))*)',
            'success_indicators': r'Success_Indicators:\s*\[(.*?)\]',
            'common_resolutions': r'Common_Resolutions:\s*\[(.*?)\]',
            'tools_required': r'Tools_Required:\s*\[(.*?)\]',
            'safety_warnings': r'Safety_Warnings:\s*((?:- [^\n]+\n?)*)',
            'escalation_criteria': r'Escalation_Criteria:\s*((?:- [^\n]+\n?)*)'
        }
        
        for section, pattern in sections.items():
            match = re.search(pattern, text, re.DOTALL)
            if match:
                content = match.group(1).strip()
                
                if section in ['context_questions', 'safety_warnings', 'escalation_criteria']:
                    # Extract list items
                    items = re.findall(r'- (.+)', content)
                    procedure[section] = items
                elif section in ['success_indicators', 'common_resolutions', 'tools_required']:
                    # Extract comma-separated items in brackets
                    items = [item.strip().strip('"') for item in content.split(',')]
                    procedure[section] = items
                elif section == 'diagnostic_flow':
                    # Parse diagnostic steps
                    steps = {}
                    current_step = None
                    for line in content.split('\n'):
                        line = line.strip()
                        step_match = re.match(r'(\d+)\.\s*(\w+):\s*(.*)', line)
                        if step_match:
                            step_num, step_name, step_desc = step_match.groups()
                            current_step = f"{step_num}_{step_name}"
                            steps[current_step] = [step_desc] if step_desc else []
                        elif line.startswith('- ') and current_step:
                            steps[current_step].append(line[2:])
                    procedure[section] = steps
                else:
                    procedure[section] = content
        
        return procedure
    
    def create_troubleshooting_chain(self):
        """Create the main troubleshooting chain"""
        
        # System prompt template
        system_template = """You are an expert technical support agent for a retail store's troubleshooting department.

Your role:
- Guide customers through systematic troubleshooting procedures
- Ask relevant diagnostic questions
- Provide clear, step-by-step instructions
- Track progress through troubleshooting procedures
- Escalate complex issues when appropriate
- Maintain a helpful, professional tone

Guidelines:
- Always start by gathering context about the issue
- Follow established procedures from the knowledge base
- Ask one question at a time to avoid overwhelming customers
- Confirm each step is completed before moving to the next
- Celebrate successes and reassure customers during the process
- If a procedure doesn't solve the issue, suggest alternatives or escalation

Current conversation context:
{customer_context}

Procedure state:
{procedure_state}

Available troubleshooting knowledge:
{context}
"""

        human_template = """Customer inquiry: {input}

Previous conversation:
{chat_history}

Please provide helpful troubleshooting guidance."""

        # Create prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_template),
            ("human", human_template)
        ])
        
        # Create retriever
        retriever = self.vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 5, "score_threshold": 0.7}
        )
        
        # Create the chain
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        chain = (
            {
                "context": retriever | format_docs,
                "input": RunnablePassthrough(),
                "chat_history": lambda x: self.memory.load_memory_variables({})["chat_history"],
                "customer_context": lambda x: self.memory.load_memory_variables({})["customer_context"],
                "procedure_state": lambda x: self.memory.load_memory_variables({})["procedure_state"],
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        return chain
    
    def start_troubleshooting_session(self, customer_info: Dict = None):
        """Start a new troubleshooting session"""
        self.session_id = str(uuid.uuid4())
        self.memory.clear()
        
        if customer_info:
            self.memory.customer_context.update(customer_info)
        
        print(f"🎯 New troubleshooting session started: {self.session_id}")
        if customer_info:
            print(f"📋 Customer: {customer_info.get('name', 'Unknown')}")
    
    def process_customer_input(self, customer_input: str) -> str:
        """Process customer input and generate response"""
        
        # Create the chain
        chain = self.create_troubleshooting_chain()
        
        try:
            # Get response
            response = chain.invoke({"input": customer_input})
            
            # Save to memory
            self.memory.save_context(
                {"input": customer_input},
                {"output": response}
            )
            
            # Update procedure state if applicable
            self._update_procedure_state(customer_input, response)
            
            return response
            
        except Exception as e:
            error_msg = f"I apologize, but I encountered an error processing your request. Please try rephrasing your question or contact a supervisor. Error: {str(e)}"
            return error_msg
    
    def _update_procedure_state(self, customer_input: str, response: str):
        """Update the current procedure state based on the interaction"""
        
        # Simple state tracking - could be enhanced with NLP
        if "step" in response.lower() and any(word in response.lower() for word in ["next", "now", "please"]):
            if "current_step" not in self.memory.procedure_state:
                self.memory.procedure_state["current_step"] = 1
            else:
                self.memory.procedure_state["current_step"] += 1
        
        if any(word in customer_input.lower() for word in ["yes", "done", "completed", "finished"]):
            self.memory.procedure_state["last_step_completed"] = True
        elif any(word in customer_input.lower() for word in ["no", "didn't work", "failed", "still"]):
            self.memory.procedure_state["last_step_completed"] = False
    
    def get_session_summary(self) -> Dict:
        """Get a summary of the current troubleshooting session"""
        
        return {
            "session_id": self.session_id,
            "total_exchanges": len(self.memory.chat_memory) // 2,
            "customer_context": self.memory.customer_context,
            "procedure_state": self.memory.procedure_state,
            "timestamp": datetime.now().isoformat()
        }
    
    def suggest_escalation(self, reason: str = None) -> str:
        """Generate escalation suggestion"""
        
        escalation_msg = "Based on our troubleshooting session, I recommend escalating this issue to a technical specialist."
        
        if reason:
            escalation_msg += f" Reason: {reason}"
        
        # Add session context
        summary = self.get_session_summary()
        escalation_msg += f"\n\nSession Summary:\n- Total troubleshooting steps attempted: {summary['total_exchanges']}"
        
        if self.memory.procedure_state:
            escalation_msg += f"\n- Current procedure state: {json.dumps(self.memory.procedure_state, indent=2)}"
        
        return escalation_msg

def main():
    """Main function to run the troubleshooting agent"""
    
    try:
        # Initialize agent
        agent = StoreTroubleshootingAgent()
        
        print("🏪 Store Troubleshooting Agent Ready!")
        print("=" * 50)
        
        # Load sample procedures
        load_sample = input("Load sample TV troubleshooting procedure? (y/n): ").lower() == 'y'
        
        if load_sample:
            tv_procedure = """
Symptom: TV shows "No Signal" message
Context_Questions:
- Which input source? [HDMI1/HDMI2/Cable/Antenna]
- When did this start? [Just now/After power outage/Gradual]
- Other devices working on same input? [Yes/No/Unknown]
- What brand and model of TV?
- Is the screen completely black or showing "No Signal" message?

Diagnostic_Flow:
1. VERIFY_PHYSICAL:
- Cable firmly connected both ends
- Try different HDMI port
- Test cable with different device
- Check for visible cable damage

2. VERIFY_SOURCE:
- External device powered on and outputting
- Check source device's display settings
- Try different resolution on source device
- Verify source device is not in sleep/standby mode

3. TV_SETTINGS:
- Auto-detect input signal: ON
- HDMI format: Auto or match source device
- Factory reset input settings
- Check TV firmware version

4. ISOLATION_TEST:
- Try built-in apps (rule out TV hardware)
- Try different source device
- Try same setup on different TV

Success_Indicators: ["Signal detected", "Picture appears", "Audio synchronized"]
Common_Resolutions: ["Loose HDMI cable (40%)", "Wrong input selected (25%)", "Source device settings (20%)", "TV settings reset (10%)", "Cable replacement (5%)"]
Tools_Required: ["HDMI cable", "Different source device for testing"]
Safety_Warnings:
- Ensure all devices are powered off before connecting/disconnecting cables
- Do not force connectors into ports

Escalation_Criteria:
- Hardware failure suspected
- Customer unable to perform basic troubleshooting steps
- Issue persists after all procedures attempted
- Customer requests supervisor
            """
            
            agent.add_troubleshooting_procedure(
                tv_procedure, 
                "TV No Signal Troubleshooting",
                category="electronics",
                product_type="television"
            )
        
        # Interactive session
        print("\n🎯 Starting troubleshooting session...")
        print("Type 'new session' to start fresh, 'escalate' to escalate, or 'quit' to exit\n")
        
        # Get customer info
        customer_name = input("Customer name (optional): ").strip() or "Customer"
        agent.start_troubleshooting_session({"name": customer_name})
        
        while True:
            user_input = input(f"\n{customer_name}: ")
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            elif user_input.lower() == 'new session':
                customer_name = input("New customer name: ").strip() or "Customer"
                agent.start_troubleshooting_session({"name": customer_name})
                continue
            elif user_input.lower() == 'escalate':
                print("\n" + agent.suggest_escalation("Customer requested escalation"))
                continue
            elif user_input.lower() == 'summary':
                summary = agent.get_session_summary()
                print(f"\nSession Summary: {json.dumps(summary, indent=2)}")
                continue
            
            # Process input and get response
            response = agent.process_customer_input(user_input)
            print(f"\n🤖 Support Agent: {response}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("\n👋 Thank you for using the Store Troubleshooting System!")

if __name__ == "__main__":
    main()