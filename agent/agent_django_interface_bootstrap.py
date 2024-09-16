import django
import os
import sys
import requests
from tools.editor import WeaveEditor

#startblock type: orientation
#timestamp 1724982545.6534579
"""
My task is to write a web app for interacting with myself that runs on localhost port 8081.
This server should be written in Django using sqlite as the backend. It serves the following functions:

- Acts as a daemon process to stop docker from closing when I completes my tasks.
- Dynamically displays the WeaveKanban as HTML lists so that the user can track my progress through the task.
- Provides a full rendering of the agent trace (event stream, in the representation 
that winds up in the context window) up to that point.
- Provides a chat window for the user and the agent to communicate with each other 
that writes into a sqlite backend.
- Provides a button to download the full contents of the working directory for 
weave-agent as a zip file.

I can edit project files using the WeaveEditor, which provides append and edit 
operations. These operations are a major improvement over the standard library 
and allow me to ergonomically modify files like views.py, urls.py, etc.

I'll start by setting up the Django project and creating the necessary models and views.
"""
#endblock
#startblock type: task_inference
#timestamp 1724982419.5427651
"""
This is the bootstrap block, which means I haven't added any tasks to the
WeaveKanban yet and will want to do so now.
"""

# New Tasks:
agent.add_task(
    "Set up Django project",
    "Set up the Django project and create the necessary models and views by writing to the files.",
    "idle"
)

agent.add_task(
    "Create WeaveKanban view",
    "Edit project files to include a view that dynamically displays the WeaveKanban as HTML lists.",
    "idle"
)

agent.add_task(
    "Create agent trace view",
    "Edit project files to create a view that provides a full rendering of the agent trace up to that point.",
    "idle"
)

agent.add_task(
    "Create chat window",
    "Edit project files to create a chat window for the user and the agent to communicate with each other that writes into a sqlite backend.",
    "idle"
)

agent.add_task(
    "Create download button",
    "Edit project files to create a button to download the full contents of the working directory for weave-agent as a zip file.",
    "idle"
)

agent.add_task(
    "Run Django server",
    "Run the Django server on localhost port 8081.",
    "idle"
)

# Updates to Existing Task
agent.current_task = agent.tasks.get_task(1)
agent.current_task.going("Setting up Django project...")
#endblock
#startblock type: action
#timestamp 1724982671.1163404
"""
I'll open the editor on the Django project and set up the necessary models and views.
"""

def setup_django_project(agent):
    # Create a new Django project
    os.system("django-admin startproject weave_agent_webapp")

    # Change directory to the project directory
    os.chdir("weave_agent_webapp")

    # Create a new Django app
    os.system("python manage.py startapp main")

    # Run migrations
    os.system("python manage.py migrate")

    # Create a superuser
    os.system("echo 'from django.contrib.auth.models import User; User.objects.create_superuser(\"admin\", \"admin@example.com\", \"password\")' | python manage.py shell")

    # Example of using WeaveEditor to edit a file
    editor = WeaveEditor(agent, "/app/weave_agent_webapp/main/views.py")
    editor.append("from django.shortcuts import render\n\n")
    editor.append("def index(request):\n    return render(request, 'index.html')\n")
    editor.close()

    return True

agent.add_action("Set up Django project", setup_django_project)
#endblock
#startblock type: expectation
#timestamp 1724982799.3917813
"""
The action should succeed if the Django project is set up correctly and the necessary models and views are created.
The action should fail if the Django project cannot be set up or if the models and views cannot be created.
"""
#endblock
#startblock type: observation_inference
#timestamp 1724982929.9047914
"""
I'll set up an observation to check the contents of the weave_agent_webapp directory.
This should let me see what its current state is.
"""

def view_django_project(agent):
    return os.listdir("/app/weave_agent_webapp")

# Add the new view
agent.add_observation_view("View Django Project Directory", view_django_project)
#endblock
#startblock type: evaluation
#timestamp 1724983062.124238

def evaluate_django_project_setup(agent):
    """Check if the Django project is set up correctly"""
    return os.path.exists("/app/weave_agent_webapp/manage.py")

def evaluate_views_py_content(agent):
    """Check if the views.py file contains the expected content"""
    with open("/app/weave_agent_webapp/main/views.py", "r") as file:
        content = file.read()
        return "def index(request):" in content

def evaluate_weavekanban_view(agent):
    """Check if the WeaveKanban view is created"""
    with open("/app/weave_agent_webapp/main/views.py", "r") as file:
        content = file.read()
        return "def weavekanban_view(request):" in content

def evaluate_agent_trace_view(agent):
    """Check if the agent trace view is created"""
    with open("/app/weave_agent_webapp/main/views.py", "r") as file:
        content = file.read()
        return "def agent_trace_view(request):" in content

def evaluate_chat_window(agent):
    """Check if the chat window is created"""
    with open("/app/weave_agent_webapp/main/views.py", "r") as file:
        content = file.read()
        return "def chat_window(request):" in content

def evaluate_download_button(agent):
    """Check if the download button is created"""
    with open("/app/weave_agent_webapp/main/views.py", "r") as file:
        content = file.read()
        return "def download_button(request):" in content

def evaluate_django_server_running(agent):
    """Check if the Django server is running"""
    try:
        response = requests.get("http://localhost:8081")
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

# Add evaluations to tasks
task1 = agent.tasks.get_task(1)
task1.add_evaluation("Evaluate Django Project Setup", evaluate_django_project_setup)
task1.add_evaluation("Evaluate views.py Content", evaluate_views_py_content)

task2 = agent.tasks.get_task(2)
task2.add_evaluation("Evaluate WeaveKanban View", evaluate_weavekanban_view)

task3 = agent.tasks.get_task(3)
task3.add_evaluation("Evaluate Agent Trace View", evaluate_agent_trace_view)

task4 = agent.tasks.get_task(4)
task4.add_evaluation("Evaluate Chat Window", evaluate_chat_window)

task5 = agent.tasks.get_task(5)
task5.add_evaluation("Evaluate Download Button", evaluate_download_button)

task6 = agent.tasks.get_task(6)
task6.add_evaluation("Evaluate Django Server Running", evaluate_django_server_running)

# Add action evaluations
def check_django_project_created(agent):
    return os.path.exists("/app/weave_agent_webapp")

agent.add_evaluation(
    "Check Django Project Created",
    check_django_project_created
)

def check_django_app_created(agent):
    return os.path.exists("/app/weave_agent_webapp/main")

agent.add_evaluation(
    "Check Django App Created",
    check_django_app_created
)

def check_views_py_edited(agent):
    with open("/app/weave_agent_webapp/main/views.py", "r") as file:
        content = file.read()
        return "def index(request):" in content

agent.add_evaluation(
    "Check views.py Edited",
    check_views_py_edited
)
#endblock
