#subagent main
#startblock type: evaluation
#index 42
#timestamp 1737490568.8009431
#time_remaining 20415.256756067276 seconds
#bm25_query type:'evaluation'  +weave-agent  +framework  +execute  +code  +agent  +subagent  +send  +message  +respond  +personal  +thought  +internal  +monologue  +interaction  +query  +observation  +callback  +tools  +discord  +client  +context  +expectation  +outcome +schema  +validation  +refine  +response +user  +message +error +execute +tick +execute-program +interaction-flow +goal-resolution +action-validation +bm25 +verify +parsed +messages +refinement  +personal-thoughts  +internal-experience +firsthand-view  +chatbot  +dialogue-loop +user-insight +query-response  +personal-reflection  +feedbac

def check_reflective_response(subagent):
    """Check if the bot's response to federovist's query is reflective and personal."""
    context = subagent.tools['discord-bot-1325039818673094739'].get_messages()
    for msg in reversed(context):
        if msg['author'] == 'weave-agent':
            bot_response = msg['content']
            break
    else:
        return 0
    reflective_keywords = ['personal', 'first-hand', 'insights', 'feelings', 'experience', 'reflection', 'thoughts', 'perspective']
    personal_keywords = ['I', 'my', 'me', "I'm"]
    reflective_count = sum((bot_response.lower().count(keyword) for keyword in reflective_keywords))
    personal_count = sum((bot_response.lower().count(keyword) for keyword in personal_keywords))
    if reflective_count > 1 and personal_count > 0:
        return 1
    return 0

self.add_evaluation('Check reflective response', check_reflective_response)
#q: Is this evaluation a good implementation of a callback to gather and evaluate the expected sensory evidence laid out in the expectation block? Yes. (73.375%)
#endblock
