from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
import os

def bootstrap_callback(agent):
    # Set up Selenium WebDriver
    driver = webdriver.Firefox()
    driver.get("https://www.google.com")

    # Set up an observation view that reads the current URL
    def read_current_url(agent):
        try:
            return driver.current_url
        except Exception as e:
            agent.add_error_block(f"Failed to read current URL: {e}")
            return ""

    agent.add_observation_view({
        'type': 'observation',
        'callback': read_current_url
    })

    # Set up an observation view that reads the page title
    def read_page_title(agent):
        try:
            return driver.title
        except Exception as e:
            agent.add_error_block(f"Failed to read page title: {e}")
            return ""

    agent.add_observation_view({
        'type': 'observation',
        'callback': read_page_title
    })

    # Set up an observation view that reads the page source and strips extraneous information
    def read_page_source(agent):
        try:
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            body_content = soup.body.get_text(separator=' ', strip=True)
            return body_content
        except Exception as e:
            agent.add_error_block(f"Failed to read page source: {e}")
            return ""

    agent.add_observation_view({
        'type': 'observation',
        'callback': read_page_source
    })

    # Set up reminders
    agent.add_reminder({
        'type': 'reminder',
        'trigger_callback': lambda agent: simple_evaluate_outputs(make_simple_score_prompt("Is the agent stuck?"), agent.context),
        'reminder_callback': lambda agent: agent.add_block({'type': 'reminder', 'message': 'The agent might be stuck. Consider re-evaluating the current approach.'}),
        'trigger_type': 'yes_no_logit',
        'question': 'Is the agent stuck?',
        'threshold': 0.8
    })

    agent.add_reminder({
        'type': 'reminder',
        'trigger_callback': lambda agent: all(os.path.exists(f"{i}.ml") for i in range(1, 26)),
        'reminder_callback': lambda agent: agent.shutdown(),
        'trigger_type': 'callback',
        'threshold': 1.0
    })

    # Set up tasks
    agent.add_task({
        'type': 'task',
        'title': 'Navigate to a search engine',
        'priority': 0,
        'parent': None,
        'children': []
    })

    agent.add_task({
        'type': 'task',
        'title': 'Search for OCaml projects',
        'priority': 1,
        'parent': 'Navigate to a search engine',
        'children': []
    })

    agent.add_task({
        'type': 'task',
        'title': 'Check licenses of OCaml projects',
        'priority': 2,
        'parent': 'Search for OCaml projects',
        'children': []
    })

    agent.add_task({
        'type': 'task',
        'title': 'Find 25 pure functions',
        'priority': 3,
        'parent': 'Check licenses of OCaml projects',
        'children': []
    })

    agent.add_task({
        'type': 'task',
        'title': 'Extract and save functions',
        'priority': 4,
        'parent': 'Find 25 pure functions',
        'children': []
    })

    agent.add_task({
        'type': 'task',
        'title': 'Shut down the agent',
        'priority': 5,
        'parent': 'Extract and save functions',
        'children': []
    })

    # Set up the initial action to navigate to a search engine
    def navigate_to_search_engine(agent):
        try:
            search_box = driver.find_element(By.NAME, "q")
            search_box.send_keys("OCaml open source projects")
            search_box.send_keys(Keys.RETURN)
        except Exception as e:
            agent.add_error_block(f"Failed to navigate to search engine: {e}")

    agent.generate_block("action", navigate_to_search_engine)

bootstrap_callback(agent)
