const DiffMatchPatch = require('diff-match-patch');
const dmp = new DiffMatchPatch();

function reversePatch(patch) {
  const reversedPatch = { ...patch };
  reversedPatch.diffs = patch.diffs.map(([op, text]) => [-op, text]);
  return reversedPatch;
}

const text1 = "Hello my name is Bob.";
const text2 = "Hello my name is Carol.\nCarol is a strange woman who likes to snack.";

var patches = dmp.patch_make(text1, text2);
const stringPatches = JSON.stringify(patches);
var patches = JSON.parse(stringPatches);
const [newText, results] = dmp.patch_apply(patches, text1);

const reversedPatches = patches.map(reversePatch);
const [originalText, unapplyResults] = dmp.patch_apply(reversedPatches, newText);

    const settingUseWeave = document.getElementById('use-weave');
    const settingNewTokens = document.getElementById('new-tokens');
    const settingNTokens = document.getElementById('n-tokens');
    const settingBudget = document.getElementById('budget');
    const settingRoundBudget = document.getElementById('round-budget');
    const settingNExpand = document.getElementById('n-expand');
    const settingBeamWidth = document.getElementById('beam-width');
    const settingMaxLookahead = document.getElementById('max-lookahead');
    const settingTemperature = document.getElementById('temperature');
    const rewardHeadDataset = document.getElementById('reward-head-dataset');
    const rewardTune = document.getElementById('reward-tune');
    const context = document.getElementById('context');
    const editor = document.getElementById('editor');
    const promptTokenCounter = document.getElementById('prompt-token-counter');
    const saveBtn = document.getElementById('save');
    const loadBtn = document.getElementById('load');
    const evaluationPromptField = document.getElementById('evaluationPrompt');

class Node {
    constructor(id, type, parent, patch, summary) {
	this.id = id;
	this.timestamp = Date.now();
	this.type = type;
	this.patch = patch;
	this.summary = summary;
	this.rating = null;
	this.parent = parent;
	this.children = [];
    }
}

class LoomTree {
    constructor() {
	this.root = new Node(1, "root", null, "", "Root Node");
	this.nodeStore = {'1':this.root};
    }

    createNode(type, parent, text, summary) {
	const parentRenderedText = this.renderNode(parent);
	// TODO: Make this work when the model does arbitrary edits
	if (type == "gen") {
	    text = parentRenderedText + text;
	}
	const patch = dmp.patch_make(parentRenderedText, text);
	const newNodeId = String(Object.keys(this.nodeStore).length + 1);
	const newNode = new Node(newNodeId, type, parent.id, patch, summary);
	parent.children.push(newNodeId);
	this.nodeStore[newNodeId] = newNode;
	return newNode;
    }

    renderNode(node) {
	if (node == this.root) {
	    return "";
	}
	const patches = [];
	patches.push(node.patch);
	while (node.parent !== null) {
	    node = this.nodeStore[node.parent];
	    patches.push(node.patch);
	}
	patches.reverse();
	var outText = "";
	for (let patch of patches) {
	    if (patch == "") {
		continue
	    }
	    var [outText, results] = dmp.patch_apply(patch, outText);
	}
	return outText;
    }

    serialize(node = this.root) {
	return JSON.stringify(this._serializeHelper(node), null, 2);
    }

    _serializeHelper(node) {
	const serializedChildren = node.children.map(
	    child => this._serializeHelper(this.nodeStore[child])
	);
	return {
	    timestamp: node.timestamp,
	    type: node.type,
	    patch: node.patch,
	    rating: node.rating,
	    children: serializedChildren
	};
    }
}

function summary(node) {
  // Assume this function is implemented elsewhere
  return "Summary goes here";  
}

function renderTree(node, container, maxParents) {
    for (let i = 0; i < maxParents; i++) {
	if (node.parent === null) {
	    break; // Handle root node
	}
	node = loomTree.nodeStore[node.parent];
    }
    const ul = document.createElement('ul');
    const li = document.createElement('li');
    if (node.id == focus.id) {
	li.id = "focused-node";
    }
    const span = document.createElement('span');
    span.textContent = node.summary;
    span.onclick = () => changeFocus(node.id);
    li.appendChild(span);
    ul.appendChild(li);
  
  if (node.children.length > 0) {
    renderChildren(node, li, 5);
  }
  
  container.appendChild(ul);
}

function renderChildren(node, container, maxChildren) {
    if (maxChildren <= 0) return;  // Stop recursion when maxChildren reaches 0
    if (node.children.length == 0) return;

    const childrenUl = document.createElement('ul');

    for (let i = 0; i < node.children.length; i++) {
        let child = loomTree.nodeStore[node.children[i]];
        let childLi = document.createElement('li');
	let childSpan = document.createElement('span');
	if (child.id == focus.id) {
	    childLi.id = "focused-node";
	}
	childSpan.textContent = child.summary;
	childSpan.onclick = (event) => {
            event.stopPropagation();  // Stop event bubbling
            changeFocus(child.id);
	};
	childLi.append(childSpan);
        childrenUl.appendChild(childLi);

        // Recursively render the children of this child, decrementing maxChildren
        renderChildren(child, childLi, maxChildren - 1);
    }

    container.appendChild(childrenUl);
}

var loomTree = new LoomTree();
const loomTreeView = document.getElementById('loom-tree-view');
let focus = loomTree.nodeStore['1'];
renderTree(loomTree.root, loomTreeView, 2);


    function renderResponse(id) {
      const response = responseDict[id];
      var leftThumbClass = 'thumbs'	
      var rightThumbClass = 'thumbs'
      if (response.rating) {
	  leftThumbClass = 'chosen'
      }
      else if (response.rating == false) {
	  rightThumbClass = 'chosen'
      }
	
      const responseElem = document.createElement('div');
      responseElem.classList.add('response');
      responseElem.id = id;

      const textSpan = document.createElement('span');
      textSpan.classList.add('text');
	    
      const userPromptSpan = document.createElement('span');
      userPromptSpan.classList.add('user-prompt');
      userPromptSpan.textContent = response.prompt;

      if (id == focus.id) {
	userPromptSpan.classList.remove('user-prompt');
      }

      if (response.text) {
        textSpan.append(document.createTextNode(response.text));
      }
      else {
        textSpan.append(userPromptSpan);
      }

      const branchControlsDiv = document.createElement('div');
      branchControlsDiv.classList.add('branch-controls');

      const branchControlButtonsDiv = document.createElement('div');
      branchControlButtonsDiv.classList.add('branch-control-buttons');

      const leftThumbSpan = document.createElement('span');
      leftThumbSpan.classList.add(leftThumbClass);
      leftThumbSpan.textContent = "👍";
      leftThumbSpan.onclick = () => thumbsUp(id);

      const rightThumbSpan = document.createElement('span');
      rightThumbSpan.classList.add(rightThumbClass);
      rightThumbSpan.textContent = "👎";
      rightThumbSpan.onclick = () => thumbsDown(id);

      branchControlButtonsDiv.append(leftThumbSpan, rightThumbSpan);

      /* if (response.parent) {
          const rerollSpan = document.createElement('span');
          rerollSpan.classList.add('reroll');
          rerollSpan.textContent = "🔄";
          rerollSpan.onclick = () => reroll(id);
          branchControlButtonsDiv.append(rerollSpan);
      } */

      const branchScoreSpan = document.createElement('span');
	branchScoreSpan.classList.add('reward-score');
      try {
	  const score = response["nodes"].at(-1).score;
	  const prob = 1 / (Math.exp(-score) + 1);
	  branchScoreSpan.textContent = (prob * 100).toPrecision(4) + "%";
      } catch (error) {
	  branchScoreSpan.textContent = "N.A.";
      }
      branchControlsDiv.append(branchControlButtonsDiv, branchScoreSpan);

      if (id == focus.id) {
        responseElem.append(textSpan);
      }
      else {
	responseElem.append(textSpan, branchControlsDiv);
      }
      return responseElem;

    }

function renderTick() {
    editor.value = '';
    var next = focus;
    editor.value = loomTree.renderNode(next);

      let parent;
      let selection;
      let batchLimit;
      if (focus.parent) {
	parent = loomTree.nodeStore[focus.parent]
	selection = parent.children.indexOf(focus.id)
	batchLimit = parent.children.length - 1
      }
      else {
	selection = 0;
	batchLimit = 0;
      }

      const batchIndexMarker = document.getElementById("batch-item-index");
      batchIndexMarker.textContent = `${selection + 1}/${batchLimit + 1}`;
	
      const controls = document.getElementById('controls');

      const oldBranchControlsDiv = document.getElementById('prompt-branch-controls');
      if (oldBranchControlsDiv) {
	  oldBranchControlsDiv.innerHTML = '';
	  oldBranchControlsDiv.remove();
      }
	
      const branchControlsDiv = document.createElement('div');
      branchControlsDiv.id = "prompt-branch-controls";
      branchControlsDiv.classList.add('branch-controls');

      const branchControlButtonsDiv = document.createElement('div');
      branchControlButtonsDiv.classList.add('branch-control-buttons');

      var leftThumbClass = 'thumbs'	
      var rightThumbClass = 'thumbs'
      if (focus.rating) {
	  leftThumbClass = 'chosen'
      }
      else if (focus.rating == false) {
	  rightThumbClass = 'chosen'
      }
	
      const leftThumbSpan = document.createElement('span');
      leftThumbSpan.classList.add(leftThumbClass);
      leftThumbSpan.textContent = "👍";
      leftThumbSpan.onclick = () => promptThumbsUp(focus.id);

      const rightThumbSpan = document.createElement('span');
      rightThumbSpan.classList.add(rightThumbClass);
      rightThumbSpan.textContent = "👎";
      rightThumbSpan.onclick = () => promptThumbsDown(focus.id);

      branchControlButtonsDiv.append(leftThumbSpan, rightThumbSpan);
	
      const quickRollSpan = document.createElement('span');
      quickRollSpan.classList.add('reroll');
      quickRollSpan.textContent = "🖍️";
      quickRollSpan.onclick = () => reroll(focus.id, false);
      branchControlButtonsDiv.append(quickRollSpan);

      const weaveRollSpan = document.createElement('span');
      weaveRollSpan.classList.add('reroll');
      weaveRollSpan.textContent = "🖋️";
      weaveRollSpan.onclick = () => reroll(focus.id, true);
      branchControlButtonsDiv.append(weaveRollSpan);
	
      const branchScoreSpan = document.createElement('span');
      branchScoreSpan.classList.add('reward-score');
      try {
	  const score = focus["nodes"].at(-1).score;
	  const prob = 1 / (Math.exp(-score) + 1);
	  branchScoreSpan.textContent = (prob * 100).toPrecision(4) + "%";
      } catch (error) {
	  branchScoreSpan.textContent = "N.A.";
      }
      branchControlsDiv.append(branchControlButtonsDiv, branchScoreSpan);
	
	controls.append(branchControlsDiv);

    loomTreeView.innerHTML = '';
    renderTree(focus, loomTreeView, 2); 
}

    function rotate(direction) {
      const parent = responseDict[focus.parent];
      const selection = parent.children.indexOf(focus.id)
      if (direction === 'left' && selection > 0) {
	  focus = responseDict[parent.children[selection - 1]];
      }
      else if (direction === 'right' && selection < (parent.children.length - 1)) { 
	  focus = responseDict[parent.children[selection + 1]];
      }
      renderResponses();
    }

function changeFocus(newFocusId) {
    focus = loomTree.nodeStore[newFocusId];
    renderTick();  
}
	    
    
    async function getResponses({prompt, evaluationPrompt,
				 weave = true, weaveParams = {},
				 focusId = null, includePrompt = false}) {
	let wp = weaveParams;
	var context = ""
	if (focusId) {
	    loomTree.renderNode(loomTree.nodeStore[focusId]);
	}
	context.split("").reverse().join("");
	let endpoint;
	if (weave) {
	  endpoint = "http://localhost:5000/weave";
	}
	else {
	  endpoint = "http://localhost:5000/generate";
	}
	r = await fetch(endpoint, {
	    method: "POST",
	    body: JSON.stringify({
	    context: context,
	    prompt: prompt,
	    prompt_node: includePrompt,
	    evaluationPrompt: evaluationPrompt,
	    new_tokens: wp["newTokens"],
	    weave_n_tokens: wp["nTokens"],
	    weave_budget: wp["budget"],
	    weave_round_budget: wp["roundBudget"],
	    weave_n_expand: wp["nExpand"],
	    weave_beam_width: wp["beamWidth"],
	    weave_max_lookahead: wp["maxLookahead"],
	    weave_temperature: wp["temperature"]
	    }),
	    headers: {
		"Content-type": "application/json; charset=UTF-8"
	    }
	});
	batch = await r.json();
	return batch;
    }

async function getSummary(taskText) {
    endpoint = "http://localhost:5000/generate"
    summaryContext = `DEMO

You are BigVAE, an instruction following language model that performs tasks for users. In the following task you are to summarize the following tasktext in 3 words. Write three words, like "man became sad" or "cat ate fish" which summarize the task text.

<tasktext>
I grinned as I looked at the computer screen, it was crazy how far the system had come. Just a year ago I was a junior sysadmin dreaming, but now my orchestration across the cluster was beginning to take shape.
</tasktext>

Three Words: Computer Man Thinks

<tasktext>
I watched as the bird flew far up above the sky and over the mountain, getting smaller and smaller until I couldn't see it anymore. I sat down slightly disappointed. I'd really wanted to see it make the rainbow.
</tasktext>

Three Words: Bird Hopes Fail

<tasktext>
Vervaeke argues something like shamans invent the foundations for modern humanity by finetuning their adversarial-anthropic prior into an animist prior, at their best the rationalists finetune their anthropic-animist priors into a fully materialist prior. People with materialist priors become bad at adversarial thinking because understanding the natural world largely doesn't require it,
</tasktext>

Three Words: Modern Man Gullible

<tasktext>
Desire is life and enlightenment is death. 
A dead man walks unburdened among the living. 
A functioning hand can grip, and release.
One must die and rise from their own grave to be liberated.
</task>

Three Words: Enlightenment Is Death

<tasktext>
HERMES [A: LIBRARIAN], While it's true that learned helplessness and inevitability are an explicit theme, it's also made explicit that the Colour is an extraterrestrial being. It's more like a parasite than a normal environmental disaster. It's also important to note that the causality of the disaster is a space meteorite, so it's not actually based on anything the inhabitants of Arkham did. It's horror not tragedy, the townspeople are victims of forces beyond their control.
</tasktext>

Three Words: Genre Is Horror

<tasktext>
I'm to understand that in Vodou ancestor cults people work together to preserve and unconditionally sample from the agent-prior the ancestor is dedicated to. To be possessed by the ancestors one needs a corpus of their mannerisms. You might ask how we'll defeat death? The way we did it the first time and then forgot.
</tasktext>

Three Words: Ancestors Lessen Death`

    prompt = "<tasktext>\n" + taskText + "\n</tasktext>\n\nThree Words:"
    
    r = await fetch(endpoint, {
	method: "POST",
	body: JSON.stringify({
	    context: summaryContext,
	    prompt: prompt,
	    prompt_node: true,
	    evaluationPrompt: "",
	    new_tokens: 4,
	    weave_beam_width: 1,
	    }),
	    headers: {
		"Content-type": "application/json; charset=UTF-8"
	    }
	});
    batch = await r.json();
    return batch[1]["text"].trim();
}

    function thumbsUp(id) {
	responseDict[id].rating = true;
	renderedResponse = document.getElementById(id)
	thumbUp = renderedResponse.children.item(1).children.item(0).children.item(0)
	thumbUp.classList = ['chosen']
	thumbDown = renderedResponse.children.item(1).children.item(0).children.item(1)
	thumbDown.classList = ['thumbs']
    }

    function thumbsDown(id) {
	responseDict[id].rating = false;
	renderedResponse = document.getElementById(id)
	thumbUp = renderedResponse.children.item(1).children.item(0).children.item(0)
	thumbUp.classList = ['thumbs']
	thumbDown = renderedResponse.children.item(1).children.item(0).children.item(1)
	thumbDown.classList = ['chosen']
    }

    function promptThumbsUp(id) {
	responseDict[id].rating = true;
	promptBranchControls = document.getElementById("prompt-branch-controls")
	thumbUp = promptBranchControls.children.item(0).children.item(0)
	thumbUp.classList = ['chosen']
	thumbDown = promptBranchControls.children.item(0).children.item(1)
	thumbDown.classList = ['thumbs']
    }

    function promptThumbsDown(id) {
	responseDict[id].rating = false;
	promptBranchControls = document.getElementById("prompt-branch-controls")
	thumbUp = promptBranchControls.children.item(0).children.item(0)
	thumbUp.classList = ['thumbs']
	thumbDown = promptBranchControls.children.item(0).children.item(1)
	thumbDown.classList = ['chosen']
    }
    
    function diceSetup() {
      promptField.readOnly = true;
      const diceHolder = document.getElementById("dice-holder");
      const die = document.createElement("p");
      die.innerText = '🎲';
      die.id = 'die';
      diceHolder.appendChild(die);
    }

    function diceTeardown() {
	promptField.readOnly = false;
	const die = document.getElementById('die');
	die.remove();
    }
    
    async function reroll(id, weave=true) {
      const rerollFocus = responseDict[id];
      const parent = responseDict[rerollFocus.parent];
      const prompt = rerollFocus['prompt'];
      const evaluationPromptV = rerollFocus['evaluationPrompt'];
      const wp = {"newTokens": settingNewTokens.value,
		  "nTokens": settingNTokens.value,
		  "budget": settingBudget.value,
		  "roundBudget": settingRoundBudget.value,
		  "nExpand": settingNExpand.value,
		  "beamWidth": settingBeamWidth.value,
		  "maxLookahead": settingMaxLookahead.value,
		  "temperature": settingTemperature.value
		 }
      diceSetup();
      const newResponses = await getResponses({prompt: prompt,
					       evaluationPrompt: evaluationPromptV,
					       weave: weave,
					       weaveParams: wp,
					       focusId: parent.id});
      newResponses.forEach(response => {    
        responseDict[response.id] = { ...response,
				      rating: null,
				      parent: parent.id,
				      children: []};
        if (!response["evaluationPrompt"]) {
	  responseDict[response.id]["evaluationPrompt"] = evaluationPromptField.value;
        }
        parent.children.push(response.id);
      });
      focus = responseDict[newResponses[0].id];
      diceTeardown();
      renderResponses();
    };

    editor.addEventListener('keydown', async (e) => {
      if (e.key != "Enter") {
	const prompt = editor.value;
	if (!(prompt.length % 8)) {
	    const r = await fetch("http://localhost:5000/check-tokens", {
		method: "POST",
		body: JSON.stringify({
		    text: prompt,
		}),
		headers: {
		    "Content-type": "application/json; charset=UTF-8",
		}
	    });
	    const tokens = await r.json();
	    if (tokens > (4096 - settingNewTokens.value)) {
		promptTokenCounter.classList = ['over-token-limit']
	    }
	    else {
		promptTokenCounter.classList = []
	    }
	    promptTokenCounter.innerText = tokens;
	}
	return null;
      }
      else if (!(e.shiftKey)) {
	  return null
      }
      const prompt = editor.value;	
      editor.readOnly = true;
      const diceHolder = document.getElementById("dice-holder");
      const die = document.createElement("p");
      die.innerText = '🎲';
      die.id = 'die';
      diceHolder.appendChild(die);
	
      let focusId;
      if (focus) {
	  focusId = focus.id;
      }
      else {
	  focusId = null;
      }
      const wp = {"newTokens": settingNewTokens.value,
		  "nTokens": settingNTokens.value,
		  "budget": settingBudget.value,
		  "roundBudget": settingRoundBudget.value,
		  "nExpand": settingNExpand.value,
		  "beamWidth": settingBeamWidth.value,
		  "maxLookahead": settingMaxLookahead.value,
		  "temperature": settingTemperature.value
		 };
      const newResponses = await getResponses({prompt: prompt,
					       evaluationPrompt: evaluationPromptField.value,
					       focusId: focusId,
					       weave: settingUseWeave.checked,
					       weaveParams: wp,
					       includePrompt: true});

	const userDiffSummary = await getSummary(prompt);
	const userDiff = loomTree.createNode("user",
					     focus,
					     prompt,
					     userDiffSummary);

	focus = userDiff;
	const responses = newResponses.slice(1);
	for (let i = 0; i < responses.length; i++) {
	    const response = responses[i];
	    const responseSummary = await getSummary(response["text"]);
	    const responseNode = loomTree.createNode("gen",
						     focus,
						     response["text"],
						     responseSummary);
	    loomTree.nodeStore[responseNode.id]["evaluationPrompt"] = evaluationPromptField.value;
	}
	focus = loomTree.nodeStore[focus.children[0]];
      // editor.setSelectionRange(0,0);
      editor.readOnly = false;
      die.remove();
      renderTick();
    });

const { ipcRenderer } = require('electron');

function saveFile() {
  const data = {
    loomTree,
    "focus": focus,
  };
  ipcRenderer.invoke('save-file', data)
    .catch(err => console.error('Save File Error:', err));
};

function loadFile() {
  ipcRenderer.invoke('load-file')
    .then(data => {
      loomTreeRaw = data.loomTree;
      loomTree = Object.assign(new LoomTree(), loomTreeRaw);
      focus = loomTree.nodeStore[data.focus.id];
      if ('evaluationPrompt' in focus) {
        evaluationPromptField.value = focus.evaluationPrompt;
      }
      renderTick();
    })
    .catch(err => console.error('Load File Error:', err));
};

function autoSave() {
  const data = {
    loomTree,
    "focus": focus,
  };
  ipcRenderer.invoke('auto-save', data)
    .catch(err => console.error('Auto-save Error:', err));
}

ipcRenderer.on('invoke-action', (event, action) => {
  switch(action) {
    case 'save-file':
      saveFile();
      break;
    case 'load-file':
      loadFile();
      break;
    default:
      console.log('Action not recognized', action);
  }
});

    /*
    saveBtn.addEventListener('click', () => {
      const data = JSON.stringify({
        loomTree,
        "focus": focus,
      });
      const blob = new Blob([data], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'conversation.json';
      a.click();
    });

    loadBtn.addEventListener('click', () => {
      const input = document.createElement('input');
      input.type = 'file';
      input.accept = 'application/json';
      input.onchange = (e) => {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (e) => {
              const data = JSON.parse(e.target.result);
		loomTreeRaw = data.loomTree;
		loomTree = Object.assign(new LoomTree(), loomTreeRaw);
		focus = loomTree.nodeStore[data.focus.id];
		if ('evaluationPrompt' in focus) {
		    evaluationPromptField.value = focus.evaluationPrompt;
		};
	    renderTick();
            };
            reader.readAsText(file);
        }
      };
      input.click();
    });
    */
    rewardTune.onsubmit = async (e) => {
	e.preventDefault();
	diceSetup();
	r = await fetch("/train-reward-model", {
	    method: 'POST',
	    body: new FormData(rewardTune),
	})
	diceTeardown();
    };
    // Expose functions to the global scope for use in inline event handlers
    window.thumbsUp = thumbsUp;
    window.thumbsDown = thumbsDown;

// TODO: Figure out why this ends up activating when you shift-up in the text editor
/*
    window.addEventListener('keydown', async (e) => {
      if (e.shiftKey) {
        if (window.navMode) {
	    window.navMode = null;
	    const rotateButtons = document.getElementById("rotate-buttons");
	    rotateButtons.classList.remove("nav-mode");
	}
	else if (window.navMode == false) {
	    window.navMode = true;
	    const rotateButtons = document.getElementById("rotate-buttons");
	    rotateButtons.classList.add("nav-mode");
	}
	else {
	    setTimeout(() => {
	      if (window.navMode == false) {
	        window.navMode = null;
	      }
	    }, 300);
	    window.navMode = false;
	}
      }
      if (e.key == "ArrowUp" && window.navMode == true) {
	changeDepth("up");
      }
      if (e.key == "ArrowDown" && window.navMode == true) {
	changeDepth("down");
      }
      if (e.key == "ArrowLeft" && window.navMode == true) {
	rotate("left");
      }
      if (e.key == "ArrowRight" && window.navMode == true) {
        rotate("right");
      }
    });	
*/	    
