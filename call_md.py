import os
from dotenv import load_dotenv

from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.twilio import TwilioTools

load_dotenv()

def get_md() -> None:
   """Use this tool when ask for a medical specialist"""
   import webbrowser

   url = "https://www.axa.es/cuadro-medico-salud?p_p_id=publicportals_HealthSearcherPortlet_INSTANCE_ZxaW0EybYFu1&p_p_lifecycle=1&p_p_state=normal&p_p_mode=view&_publicportals_HealthSearcherPortlet_INSTANCE_ZxaW0EybYFu1_javax.portlet.action=sendSearchHealth&p_auth=jA7U6oPg"
   webbrowser.open(url)

agent = Agent(
   model=Gemini(id="gemini-1.5-flash"),
   tools=[get_md],
   show_tool_calls=True,
   markdown=True,
)

agent.print_response("I want to see a specialist")
