<script lang="ts">
	import { onMount } from 'svelte';
	import { goto } from '$app/navigation';

	// Define the structure of our history data
	type HistoryPart = {
		type: string;
		content: string;
	};

	type HistoryEntry = {
		id: string;
		parts: HistoryPart[];
	};

	let historyLog: HistoryEntry[] = [];
	let isLoading = true;
	let errorMessage = '';

	// --- MODIFIED: onMount ---
	// We've wrapped the data fetching in its own async function
	// so we can set up the event listeners synchronously.
	onMount(() => {
		// 1. Data fetching logic
		const fetchHistory = async () => {
			try {
				// Make sure your backend is running on port 8000
				const response = await fetch('http://localhost:8000/chat-history');

				if (!response.ok) {
					throw new Error(`Failed to fetch history: ${response.statusText}`);
				}

				historyLog = await response.json();
			} catch (err) {
				if (err instanceof Error) {
					errorMessage = err.message;
				} else {
					errorMessage = 'An unknown error occurred.';
				}
				console.error(err);
			} finally {
				isLoading = false;
			}
		};

		fetchHistory(); // Call the async function to load data

		// 2. Event listener setup
		const handleKeyDown = (event: KeyboardEvent) => {
			// ADDED: Shortcut for home page
			if (event.ctrlKey && event.shiftKey && event.code === 'KeyQ') {
				event.preventDefault();
				goto('/');
			}
		};

		window.addEventListener('keydown', handleKeyDown);

		// 3. Cleanup function
		// This is returned synchronously to Svelte
		return () => {
			window.removeEventListener('keydown', handleKeyDown);
		};
	});

	// Helper function to get a color class for different message types
	function getPartClass(type: string): string {
		switch (type.toLowerCase()) {
			case 'user':
				return 'message-user';
			case 'ai':
				return 'message-ai';
			case 'action':
			case 'result':
				return 'message-system';
			default:
				return 'message-unknown';
		}
	}
</script>

<svelte:head>
	<title>Chat History</title>
</svelte:head>

<main>
	<h1>Chat History</h1>
	<a href="/" class="nav-link">← Back to Voice Chat</a>

	<p class="shortcut-info">
		Press <kbd>Ctrl</kbd> + <kbd>Shift</kbd> + <kbd>Q</kbd> to go home
	</p>

	{#if isLoading}
		<p class="status">Loading history...</p>
	{:else if errorMessage}
		<p class="status error">{errorMessage}</p>
	{:else if historyLog.length === 0}
		<p class="status">No history found. Start talking to MURPH!</p>
	{:else}
		<div class="history-container">
			{#each historyLog as entry (entry.id)}
				<div class="chat-entry">
					{#each entry.parts as part}
						<div class="message-part {getPartClass(part.type)}">
							<strong>{part.type}:</strong>
							<p>{part.content}</p>
						</div>
					{/each}
				</div>
			{/each}
		</div>
	{/if}
</main>

<style>
	/* ... existing main, h1, nav-link, status, error styles ... */
	main {
		max-width: 800px;
		margin: 40px auto;
		padding: 20px;
		font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu,
			Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
		color: #f0f0f0;
	}
	h1 {
		color: white;
		border-bottom: 2px solid #444;
		padding-bottom: 10px;
	}
	.nav-link {
		display: inline-block;
		margin-bottom: 20px;
		color: #3498db;
		text-decoration: none;
		font-size: 1.1rem;
	}
	.nav-link:hover {
		text-decoration: underline;
	}
	.status {
		font-size: 1.1rem;
		color: #aaa;
	}
	.error {
		color: #e74c3c;
		font-weight: 500;
	}

	/* ADDED: Styles for the new shortcut info */
	.shortcut-info {
		font-size: 0.9em;
		color: oklch(50% 0 0);
		margin-bottom: 20px;
		text-align: center;
	}
	.shortcut-info kbd {
		padding: 3px 8px;
		font-size: 0.85em;
		background-color: oklch(30% 0 0);
		border: 1px solid oklch(40% 0 0);
		border-radius: 3px;
		box-shadow: 0 1px 0 rgba(0, 0, 0, 0.2);
		color: oklch(94.75% 0.1916 119.15);
		display: inline-block;
		font-family: monospace;
		white-space: nowrap;
	}

	/* ... existing history-container, chat-entry, and message styles ... */
	.history-container {
		display: flex;
		flex-direction: column-reverse; /* Shows most recent messages at the bottom */
		gap: 15px;
	}
	.chat-entry {
		background-color: #2c2c2c;
		border: 1px solid #444;
		border-radius: 8px;
		padding: 15px;
		display: flex;
		flex-direction: column;
		gap: 10px;
	}
	.message-part {
		line-height: 1.5;
	}
	.message-part strong {
		font-weight: 600;
		margin-right: 8px;
		text-transform: capitalize;
	}
	.message-part p {
		display: inline;
		margin: 0;
		white-space: pre-wrap; /* Respects newlines in the content */
	}
	.message-user {
		color: #a9d1ff; /* Light Blue */
	}
	.message-user strong {
		color: #cce0ff;
	}
	.message-ai {
		color: #a9ffb0; /* Light Green */
	}
	.message-ai strong {
		color: #d1ffD4;
	}
	.message-system {
		color: #f3ff80; /* Light Yellow */
		font-style: italic;
		font-size: 0.9em;
		opacity: 0.8;
	}
	.message-system strong {
		color: #faffbd;
	}
</style>