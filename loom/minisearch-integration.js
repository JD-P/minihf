// Add this to the top of renderer.js after the existing imports
// Note: You'll need to include MiniSearch via CDN or npm install

// Global search index variable
let searchIndex = null;
let searchResultsMode = false;
let currentSearchQuery = '';

// Initialize search index with MiniSearch
function initializeSearchIndex() {
    // Create MiniSearch index - much simpler than Lunr!
    searchIndex = new MiniSearch({
        fields: ['content', 'summary', 'type'], // fields to index for full-text search
        storeFields: ['content', 'summary', 'type', 'timestamp', 'fullContent'], // fields to return with search results
        searchOptions: {
            boost: { 
                content: 3,    // Patch content gets highest boost
                summary: 2,    // Summary gets medium boost
                type: 1        // Node type gets lower boost
            },
            prefix: true,      // Enable prefix search (great for real-time search)
            fuzzy: 0.2        // Enable fuzzy search for typos
        }
    });
    
    // Add all existing nodes to the index
    Object.keys(loomTree.nodeStore).forEach(nodeId => {
        addNodeToSearchIndex(loomTree.nodeStore[nodeId]);
    });
    
    console.log(`MiniSearch index initialized with ${Object.keys(loomTree.nodeStore).length} nodes`);
}

// Extract meaningful content from diff patches
function extractPatchContent(patch) {
    if (!patch || typeof patch === 'string') {
        return patch || '';
    }
    
    try {
        // If patch is a DiffMatchPatch patch array
        if (Array.isArray(patch)) {
            return patch.map(p => {
                if (p.diffs) {
                    return p.diffs
                        .filter(diff => diff[0] === 1) // Only get insertions (new content)
                        .map(diff => diff[1])
                        .join(' ');
                }
                return '';
            }).join(' ');
        }
        
        return '';
    } catch (error) {
        console.warn('Error extracting patch content:', error);
        return '';
    }
}

// Add a single node to the search index
function addNodeToSearchIndex(node) {
    if (!searchIndex || !node) return;
    
    const patchContent = extractPatchContent(node.patch);
    if (!patchContent.trim() && !node.summary) return; // Skip nodes with no searchable content
    
    try {
        searchIndex.add({
            id: node.id,
            content: patchContent,
            summary: node.summary || '',
            type: node.type,
            timestamp: node.timestamp,
            fullContent: loomTree.renderNode(node) // Store for display
        });
    } catch (error) {
        console.warn('Error adding node to search index:', error);
    }
}

// Remove a node from the search index
function removeNodeFromSearchIndex(node) {
    if (!searchIndex) return;
    
    try {
        searchIndex.remove(node);
    } catch (error) {
        console.warn('Error removing node from search index:', error);
    }
}

// Update a node in the search index
function updateNodeInSearchIndex(node) {
    if (!searchIndex || !node) return;
    
    try {
	searchIndex.replace({
            id: node.id,
            content: extractPatchContent(node.patch),
            summary: node.summary || '',
	    type: node.type,
	    timestamp: node.timestamp,
	    fullContent: loomTree.renderNode(node),
        });
    } catch (error) {
        console.warn('Error updating node in search index:', error);
    }
}

// Perform search with MiniSearch
function performSearch(query) {
    if (!searchIndex || !query.trim()) {
        return [];
    }
    
    try {
        const results = searchIndex.search(query, {
            // MiniSearch allows per-query options
            boost: { 
                content: 3, 
                summary: 2, 
                type: 1 
            },
            prefix: true,
            fuzzy: 0.2
        });
        
        return results.map(result => {
            const node = loomTree.nodeStore[result.id];
            return {
                node: node,
                score: result.score,
                highlightedContent: highlightText(
                    result.content || extractPatchContent(node.patch), 
                    query
                ),
                highlightedSummary: highlightText(result.summary || '', query)
            };
        });
    } catch (error) {
        console.warn('Search error:', error);
        return [];
    }
}

// Get search suggestions (bonus feature!)
function getSearchSuggestions(query) {
    if (!searchIndex || !query.trim()) {
        return [];
    }
    
    try {
        return searchIndex.autoSuggest(query, {
            fuzzy: 0.3,
            prefix: true
        });
    } catch (error) {
        console.warn('Suggestion error:', error);
        return [];
    }
}

// Simple text highlighting function
function highlightText(text, query) {
    if (!query.trim() || !text) return text;
    
    const words = query.toLowerCase().split(/\s+/);
    let highlighted = text;
    
    words.forEach(word => {
        const regex = new RegExp(`(${escapeRegex(word)})`, 'gi');
        highlighted = highlighted.replace(regex, '<mark>$1</mark>');
    });
    
    return highlighted;
}

function escapeRegex(string) {
    return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

// Render search results instead of tree
function renderSearchResults(query, container) {
    const results = performSearch(query);
    container.innerHTML = '';
    
    if (results.length === 0) {
        const noResults = document.createElement('div');
        noResults.style.cssText = 'padding: 10px; color: #666; font-style: italic;';
        noResults.textContent = 'No results found';
        container.appendChild(noResults);
        return;
    }
    
    const resultsContainer = document.createElement('div');
    resultsContainer.style.cssText = 'max-height: 400px; overflow-y: auto;';
    
    results.forEach(result => {
        const resultItem = document.createElement('div');
        resultItem.style.cssText = `
            border: 1px solid #ddd;
            margin: 5px 0;
            padding: 8px;
            cursor: pointer;
            border-radius: 3px;
            background: white;
        `;
        
        // Add hover effect
        resultItem.addEventListener('mouseenter', () => {
            resultItem.style.backgroundColor = '#f0f0f0';
        });
        resultItem.addEventListener('mouseleave', () => {
            resultItem.style.backgroundColor = 'white';
        });
        
        const header = document.createElement('div');
        header.style.cssText = 'font-weight: bold; color: #333; margin-bottom: 4px;';
        header.innerHTML = `${result.node.type.toUpperCase()} - ${result.highlightedSummary || result.node.summary}`;
        
        const content = document.createElement('div');
        content.style.cssText = 'font-size: 0.9em; color: #666; line-height: 1.3;';
        const truncatedContent = result.highlightedContent.substring(0, 150);
        content.innerHTML = truncatedContent + (truncatedContent.length < result.highlightedContent.length ? '...' : '');
        
        const meta = document.createElement('div');
        meta.style.cssText = 'font-size: 0.8em; color: #999; margin-top: 4px;';

        // Format timestamp in user's timezone
        const formattedDate = new Date(result.node.timestamp).toLocaleString('en-US', {
            year: 'numeric',
            month: 'short', 
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        });
        
        meta.innerHTML = `
            <div>Score: ${result.score.toFixed(3)} | ID: ${result.node.id}</div>
            <div style="margin-top: 2px; color: #999;">${formattedDate}</div>
        `;
	
        resultItem.appendChild(header);
        resultItem.appendChild(content);
        resultItem.appendChild(meta);
        
        // Click to focus on node
        resultItem.onclick = () => {
            currentSearchQuery = '';
            searchResultsMode = false;
            document.getElementById('search-input').value = '';
            changeFocus(result.node.id);
        };
        
        resultsContainer.appendChild(resultItem);
    });
    
    container.appendChild(resultsContainer);
}

// Modified renderTree function to support search mode
function renderTreeOrSearch(node, container, maxParents) {
    if (searchResultsMode && currentSearchQuery.trim()) {
        renderSearchResults(currentSearchQuery, container);
        return;
    }
    
    // Original renderTree logic
    for (let i = 0; i < maxParents; i++) {
        if (node.parent === null) {
            break;
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

// Override the original createNode to update search index
const originalCreateNode = LoomTree.prototype.createNode;
LoomTree.prototype.createNode = function(type, parent, text, summary) {
    const node = originalCreateNode.call(this, type, parent, text, summary);
    
    // Add to search index immediately - MiniSearch handles this efficiently!
    if (searchIndex) {
        addNodeToSearchIndex(node);
    }
    
    return node;
};

// Override the original updateNode to update search index
const originalUpdateNode = LoomTree.prototype.updateNode;
LoomTree.prototype.updateNode = function(node, text, summary) {
    originalUpdateNode.call(this, node, text, summary);
    
    // Update in search index - MiniSearch makes this easy
    if (searchIndex) {
        updateNodeInSearchIndex(node);
    }
};

// Create search bar HTML with suggestions
function createSearchBar() {
    const searchContainer = document.createElement('div');
    searchContainer.style.cssText = `
        margin-bottom: 10px;
        padding: 0;
        position: relative;
        width: 100%;
        box-sizing: border-box;
    `;
    
    const searchInput = document.createElement('input');
    searchInput.type = 'text';
    searchInput.id = 'search-input';
    searchInput.placeholder = 'Search nodes...';
    searchInput.style.cssText = `
        width: 100%;
        padding: 8px;
        border: 1px solid #ddd;
        border-radius: 3px;
        font-size: 14px;
        box-sizing: border-box;
    `;
    
    // Suggestions dropdown
    const suggestionsContainer = document.createElement('div');
    suggestionsContainer.id = 'search-suggestions';
    suggestionsContainer.style.cssText = `
        position: absolute;
        top: 100%;
        left: 0;
        right: 0;
        background: white;
        border: 1px solid #ddd;
        border-top: none;
        border-radius: 0 0 3px 3px;
        max-height: 150px;
        overflow-y: auto;
        z-index: 1000;
        display: none;
        box-sizing: border-box;
    `;
    
    // Search as you type with debouncing
    let searchTimeout;
    searchInput.addEventListener('input', (e) => {
        clearTimeout(searchTimeout);
        const query = e.target.value;
        
        searchTimeout = setTimeout(() => {
            currentSearchQuery = query;
            searchResultsMode = query.trim().length > 0;
            
            const treeView = document.getElementById('loom-tree-view');
            treeView.innerHTML = '';
            
            if (searchResultsMode) {
                renderSearchResults(query, treeView);
                showSuggestions(query, suggestionsContainer);
            } else {
                renderTree(focus, treeView, 2);
                hideSuggestions(suggestionsContainer);
            }
        }, 200); // 200ms debounce
    });
    
    // Handle suggestion clicks and keyboard navigation
    let suggestionIndex = -1;
    searchInput.addEventListener('keydown', (e) => {
        const suggestions = suggestionsContainer.querySelectorAll('.search-suggestion');
        
        if (e.key === 'Escape') {
            e.target.value = '';
            currentSearchQuery = '';
            searchResultsMode = false;
            hideSuggestions(suggestionsContainer);
            const treeView = document.getElementById('loom-tree-view');
            treeView.innerHTML = '';
            renderTree(focus, treeView, 2);
        } else if (e.key === 'ArrowDown') {
            e.preventDefault();
            suggestionIndex = Math.min(suggestionIndex + 1, suggestions.length - 1);
            updateSuggestionHighlight(suggestions, suggestionIndex);
        } else if (e.key === 'ArrowUp') {
            e.preventDefault();
            suggestionIndex = Math.max(suggestionIndex - 1, -1);
            updateSuggestionHighlight(suggestions, suggestionIndex);
        } else if (e.key === 'Enter' && suggestionIndex >= 0) {
            e.preventDefault();
            const selectedSuggestion = suggestions[suggestionIndex];
            if (selectedSuggestion) {
                searchInput.value = selectedSuggestion.textContent;
                searchInput.dispatchEvent(new Event('input'));
                hideSuggestions(suggestionsContainer);
            }
        }
    });
    
    // Hide suggestions when clicking outside
    document.addEventListener('click', (e) => {
        if (!searchContainer.contains(e.target)) {
            hideSuggestions(suggestionsContainer);
        }
    });
    
    searchContainer.appendChild(searchInput);
    searchContainer.appendChild(suggestionsContainer);
    return searchContainer;
}

// Show search suggestions
function showSuggestions(query, container) {
    if (!query.trim()) {
        hideSuggestions(container);
        return;
    }
    
    const suggestions = getSearchSuggestions(query);
    if (suggestions.length === 0) {
        hideSuggestions(container);
        return;
    }
    
    container.innerHTML = '';
    suggestions.slice(0, 5).forEach(suggestion => { // Limit to 5 suggestions
        const item = document.createElement('div');
        item.className = 'search-suggestion';
        item.textContent = suggestion.suggestion;
        item.style.cssText = `
            padding: 8px;
            cursor: pointer;
            border-bottom: 1px solid #eee;
        `;
        
        item.addEventListener('mouseenter', () => {
            item.style.backgroundColor = '#f0f0f0';
        });
        item.addEventListener('mouseleave', () => {
            item.style.backgroundColor = 'white';
        });
        item.addEventListener('click', () => {
            document.getElementById('search-input').value = suggestion.suggestion;
            document.getElementById('search-input').dispatchEvent(new Event('input'));
            hideSuggestions(container);
        });
        
        container.appendChild(item);
    });
    
    container.style.display = 'block';
}

// Hide suggestions
function hideSuggestions(container) {
    container.style.display = 'none';
}

// Update suggestion highlight for keyboard navigation
function updateSuggestionHighlight(suggestions, index) {
    suggestions.forEach((item, i) => {
        if (i === index) {
            item.style.backgroundColor = '#e0e0e0';
        } else {
            item.style.backgroundColor = 'white';
        }
    });
}

// Insert search bar into the DOM
function insertSearchBar() {
    const loomTreeView = document.getElementById('loom-tree-view');
    
    // Create a wrapper container for the tree area if it doesn't exist
    let treeContainer = document.getElementById('tree-container');
    if (!treeContainer) {
        treeContainer = document.createElement('div');
        treeContainer.id = 'tree-container';
        treeContainer.style.cssText = `
            display: flex;
            flex-direction: column;
            width: 300px;
            margin-left: 2em;
            margin-top: 1em;
        `;
        
        // Wrap the existing tree view
        loomTreeView.parentNode.insertBefore(treeContainer, loomTreeView);
        treeContainer.appendChild(loomTreeView);
        
        // Remove the original margin from tree view since container handles it
        loomTreeView.style.marginLeft = '0';
    }
    
    // Insert search bar at the top of the tree container
    const searchBar = createSearchBar();
    treeContainer.insertBefore(searchBar, loomTreeView);
}

// Modified renderTick to use new render function
const originalRenderTick = renderTick;
window.renderTick = function() {
    // Call most of the original renderTick logic
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
  
    if (focus.type === "gen") {
        const rewriteButton = document.createElement("span");
        rewriteButton.id = "rewrite-button";
        rewriteButton.textContent = "💬";
        rewriteButton.onclick = () => promptRewriteNode(focus.id);

        branchControlButtonsDiv.append(rewriteButton);
    }
    const quickRollSpan = document.createElement('span');
    quickRollSpan.classList.add('reroll');
    quickRollSpan.textContent = "🖋️";
    quickRollSpan.onclick = () => reroll(focus.id, false);
    branchControlButtonsDiv.append(quickRollSpan);

    if (focus.type === "weave") {
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
    }
    else {
        branchControlsDiv.append(branchControlButtonsDiv);
    }
        
    controls.append(branchControlsDiv);

    focus.read = true;
    const loomTreeView = document.getElementById('loom-tree-view');
    loomTreeView.innerHTML = '';
    renderTreeOrSearch(focus, loomTreeView, 2); // Use our new function
    errorMessage.textContent = "";
    updateCounterDisplay(editor.value);
};

// Modified loadFile function to rebuild search index
const originalLoadFile = loadFile;
window.loadFile = async function() {
    await originalLoadFile();
    initializeSearchIndex();
};

// Initialize everything when the page loads
document.addEventListener('DOMContentLoaded', function() {
    // Insert search bar
    insertSearchBar();
    
    // Initialize search index
    initializeSearchIndex();
    
    console.log('MiniLoom MiniSearch integration loaded');
});

// Export functions for debugging
window.searchFunctions = {
    performSearch,
    getSearchSuggestions,
    addNodeToSearchIndex,
    removeNodeFromSearchIndex,
    updateNodeInSearchIndex,
    extractPatchContent
};
