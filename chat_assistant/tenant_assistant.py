# tenant_assistant.py
import json
import os
import traceback
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup

# Load environment variables
load_dotenv()


class TenantConfig:
    """Manages tenant configuration information"""

    def __init__(self, tenant_info_path="tenant_info.json"):
        self.tenant_info_path = tenant_info_path
        self.tenants = self._load_tenant_info()

    def _load_tenant_info(self):
        """Load tenant information from JSON file"""
        if not os.path.exists(self.tenant_info_path):
            # Create sample tenant info if it doesn't exist
            self._create_sample_tenant_info()

        with open(self.tenant_info_path, 'r') as f:
            return json.load(f)

    def _create_sample_tenant_info(self):
        """Create sample tenant info for demonstration"""
        sample_tenants = {
            "acme_corp": {
                "name": "Acme Corporation",
                "password": "acme123",
                "doc_urls": [
                    "https://example.com/about",
                    "https://example.com/products"
                ]
            },
            "globex_industries": {
                "name": "Globex Industries",
                "password": "globex123",
                "doc_urls": [
                    "https://example.com/about",
                    "https://example.com/services"
                ]
            },
            "wayne_enterprises": {
                "name": "Wayne Enterprises",
                "password": "wayne123",
                "doc_urls": [
                    "https://example.com/about",
                    "https://example.com/contact"
                ]
            }
        }

        with open(self.tenant_info_path, 'w') as f:
            json.dump(sample_tenants, f, indent=2)

    def get_tenant_info(self, tenant_id):
        """Get information for a specific tenant"""
        if tenant_id not in self.tenants:
            raise ValueError(f"Tenant {tenant_id} not found")

        return self.tenants[tenant_id]

    def verify_tenant_password(self, tenant_id, password):
        """Verify if the provided password matches the tenant's password"""
        if tenant_id not in self.tenants:
            return False

        tenant_password = self.tenants[tenant_id].get("password", None)
        if tenant_password is None:
            # If no password is set, allow access (backward compatibility)
            return True

        return password == tenant_password

class WebChatbot:
    """Chatbot that uses web browsing to respond to queries"""

    def __init__(self, tenant_id):
        # Load tenant configuration
        self.tenant_config = TenantConfig()
        self.tenant_id = tenant_id
        self.tenant_info = self.tenant_config.get_tenant_info(tenant_id)

        # Initialize LLM
        try:
            self.llm = ChatOpenAI(temperature=0.2)
            print(f"Successfully initialized ChatOpenAI for tenant {tenant_id}")
        except Exception as e:
            print(f"Error initializing ChatOpenAI: {str(e)}")
            raise

        # Initialize chat history
        self.chat_history = []

        # Setup agents
        self.setup_agents()

    def setup_agents(self):
        """Set up the agents needed for the chatbot"""
        tenant_name = self.tenant_info["name"]
        print(f"Setting up agents for {tenant_name}")

        try:
            # Create the browser agent
            self.browser_agent = Agent(
                role="Web Researcher",
                goal=f"Search for information related to {tenant_name}",
                backstory=f"You are an expert web researcher who helps find information about {tenant_name}.",
                verbose=True,
                allow_delegation=False,
                llm=self.llm
            )

            # Create the chat agent
            self.chat_agent = Agent(
                role="Customer Support Representative",
                goal=f"Provide helpful support to {tenant_name}'s customers",
                backstory=f"You are a customer support representative for {tenant_name}.",
                verbose=True,
                allow_delegation=True,
                llm=self.llm
            )

            print("Agents set up successfully")

        except Exception as e:
            print(f"Error setting up agents: {str(e)}")
            traceback.print_exc()
            raise

    def _fetch_web_content(self, url):
        """Fetch content from a webpage using requests and BeautifulSoup"""
        print(f"Fetching content from: {url}")

        try:
            # Use requests to get the page content
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()  # Raise an exception for 4XX/5XX responses

            # Parse with BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract text content, removing scripts, styles, etc.
            for script in soup(["script", "style", "meta", "noscript"]):
                script.extract()

            text = soup.get_text(separator='\n')

            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)

            print(f"Successfully fetched {len(text)} characters from {url}")
            return text

        except requests.exceptions.RequestException as e:
            print(f"Request error for {url}: {str(e)}")
            return f"Error fetching content from {url}: {str(e)}"
        except Exception as e:
            print(f"Unexpected error fetching {url}: {str(e)}")
            traceback.print_exc()
            return f"Error processing content from {url}: {str(e)}"

    def _search_web(self, query):
        """Search tenant-specific web pages for information"""
        results = []

        print(f"Searching web pages for query: '{query}'")
        print(f"URLs to search: {self.tenant_info['doc_urls']}")

        for url in self.tenant_info["doc_urls"]:
            # Fetch the content
            content = self._fetch_web_content(url)

            # Check if content is an error message
            if content.startswith("Error"):
                results.append(content)
                continue

            # Process content into chunks for searching
            try:
                # Split text into paragraphs
                paragraphs = [p for p in content.split('\n\n') if p.strip()]

                # Simple text search
                matched_paragraphs = []
                for i, paragraph in enumerate(paragraphs):
                    if query.lower() in paragraph.lower():
                        matched_paragraphs.append(f"Paragraph {i}: {paragraph}")

                if matched_paragraphs:
                    print(f"Found {len(matched_paragraphs)} matching paragraphs in {url}")
                    results.append(
                        f"From {url}:\n" + "\n\n".join(matched_paragraphs[:3]))  # Limit to 3 paragraphs per URL
                else:
                    # If no exact match, include first paragraph as general information
                    print(f"No exact matches in {url}, including general information")
                    if paragraphs:
                        intro = paragraphs[0]
                        # If first paragraph is too short, include a few more
                        if len(intro) < 200 and len(paragraphs) > 1:
                            intro = paragraphs[0] + "\n\n" + paragraphs[1]
                        results.append(f"General information from {url}:\n{intro}")
            except Exception as e:
                error_msg = f"Error processing content from {url}: {str(e)}"
                print(error_msg)
                traceback.print_exc()
                results.append(error_msg)

        if not results:
            print("No relevant information found on any page")
            return "No relevant information found on the specified web pages."

        print(f"Search completed. Found information from {len(results)} pages")
        return "\n\n".join(results)

    def create_research_task(self, query):
        """Create research task for the browser agent"""
        # Search web first
        try:
            search_results = self._search_web(query)
            print("Web search completed successfully")
        except Exception as e:
            error_message = f"Error searching web: {str(e)}"
            print(error_message)
            traceback.print_exc()
            search_results = f"Unable to retrieve information from the websites due to technical issues: {error_message}. The sites may be unavailable or blocking automated access."

        # Create the research task with the search results
        try:
            task = Task(
                description=f"""
                Research the following customer query:
                "{query}"

                Focus on information related to {self.tenant_info['name']}.
                Use this information from our web search:
                {search_results}

                Provide relevant and trustworthy information. If the web search encountered errors,
                acknowledge this in your response and provide general assistance if possible.
                """,
                agent=self.browser_agent,
                expected_output="Relevant information from the web"
            )
            return task
        except Exception as e:
            print(f"Error creating research task: {str(e)}")
            traceback.print_exc()
            raise

    def create_response_task(self, query, research_results):
        """Create response task for the chat agent"""
        try:
            # Include chat history in context
            history_context = ""
            if self.chat_history:
                history_entries = []
                for msg in self.chat_history:
                    history_entries.append(f"User: {msg[0]}\nAssistant: {msg[1]}")
                history_context = "\n\n".join(history_entries)

            return Task(
                description=f"""
                You are a customer support assistant for {self.tenant_info['name']}.

                Respond to the following customer query:
                "{query}"

                Use this information from our web research:
                {research_results}

                Previous conversation history:
                {history_context}

                Be polite, helpful, and accurate in your response.
                Only provide information that is relevant to {self.tenant_info['name']}.
                If you don't know the answer, say so honestly.
                If there were technical issues accessing the website, let the user know but try to be as helpful as possible.
                """,
                agent=self.chat_agent,
                expected_output="A polite and informative response to the customer's query"
            )
        except Exception as e:
            print(f"Error creating response task: {str(e)}")
            traceback.print_exc()
            raise

    def process_query(self, query):
        """Process a user query and return a response"""
        print(f"\n--- Processing query: '{query}' ---")

        try:
            # Create research task
            research_task = self.create_research_task(query)

            # Create research crew
            research_crew = Crew(
                agents=[self.browser_agent],
                tasks=[research_task],
                process=Process.sequential,
                verbose=True
            )

            # Execute research
            print("Starting research phase...")
            research_results = research_crew.kickoff()
            print("Research phase completed")

            # Extract the string result from CrewOutput
            if hasattr(research_results, 'raw_output'):
                research_results_str = research_results.raw_output
            else:
                research_results_str = str(research_results)

            # Create response task
            response_task = self.create_response_task(query, research_results_str)

            # Create response crew
            response_crew = Crew(
                agents=[self.chat_agent],
                tasks=[response_task],
                process=Process.sequential,
                verbose=True
            )

            # Execute response
            print("Starting response phase...")
            response_output = response_crew.kickoff()
            print("Response phase completed")

            # Extract the string result from CrewOutput
            if hasattr(response_output, 'raw_output'):
                response_text = response_output.raw_output
            else:
                response_text = str(response_output)

            # Clean up response text if needed
            if "## Final Answer:" in response_text:
                # Extract just the final answer part
                response_text = response_text.split("## Final Answer:")[1].strip()

            # Update chat history
            self.chat_history.append((query, response_text))

            print(f"--- Query processing complete ---\n")
            return response_text

        except Exception as e:
            error_message = f"Error processing query: {str(e)}"
            print(error_message)
            traceback.print_exc()

            # Create a fallback response
            fallback_response = f"""
            I apologize, but I'm currently experiencing technical difficulties while trying to access information about {self.tenant_info['name']}.

            The specific error was: {str(e)}

            If you have an urgent question, you may want to try again later or contact {self.tenant_info['name']} directly through their official channels.

            Is there anything else I can help you with?
            """

            # Update chat history with the fallback response
            self.chat_history.append((query, fallback_response))

            return fallback_response


# Simple demo application for command-line testing
def main():
    # Initialize with a tenant
    tenant_id = input("Enter tenant ID (or leave blank for default 'acme_corp'): ").strip() or "acme_corp"
    try:
        chatbot = WebChatbot(tenant_id)

        print(f"Chatbot initialized for {chatbot.tenant_info['name']}")
        print(f"URLs configured: {chatbot.tenant_info['doc_urls']}")
        print("Type 'exit' to quit")

        while True:
            query = input("\nUser: ")
            if query.lower() == 'exit':
                break

            response = chatbot.process_query(query)
            print(f"\nAssistant: {response}")
    except Exception as e:
        print(f"Error initializing chatbot: {str(e)}")
        traceback.print_exc()


if __name__ == "__main__":
    main()