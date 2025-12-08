<script lang="ts">
	import { onMount } from 'svelte';
	import Clock from '$lib/components/Clock.svelte';
	import TextDisplay from '$lib/components/TextDisplay.svelte';
	import { now, equal } from '$lib/utilities/time';
	import { goto } from '$app/navigation';

	let time = now();
	let showMurph = false;
	let isRecording = false;
	let isSpeaking = false;
	let spokenText = '';
	let visibleChars = 0;
	let mediaRecorder: MediaRecorder | null = null;
	let audioChunks: Blob[] = [];
	let animationInterval: ReturnType<typeof setInterval> | null = null;
	let currentAudio: HTMLAudioElement | null = null;
	let isProcessing = false; // New flag for transition state
	let processingAnimationInterval: ReturnType<typeof setInterval> | null = null;
	let processingTime = { hours: 'MU', minutes: 'RP', seconds: 'H!' };

	const MAX_WORDS_PER_SCREEN = 15;

	// Characters to cycle through for processing animation
	const RANDOM_CHARS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!?.,';
	
	function getRandomChar() {
		return RANDOM_CHARS[Math.floor(Math.random() * RANDOM_CHARS.length)];
	}
	
	function animateProcessing() {
		processingAnimationInterval = setInterval(() => {
			processingTime = {
				hours: getRandomChar() + getRandomChar(),
				minutes: getRandomChar() + getRandomChar(),
				seconds: getRandomChar() + getRandomChar()
			};
		}, 500); // Change characters every 100ms
	}
	
	function stopProcessingAnimation() {
		if (processingAnimationInterval) {
			clearInterval(processingAnimationInterval);
			processingAnimationInterval = null;
		}
	}

	onMount(() => {
		const clockInterval = setInterval(() => {
			if (!showMurph && !isSpeaking) {
				const newTime = now();
				if (!equal(newTime, time)) {
					time = newTime;
				}
			}
		}, 100);

		const handleKeyDown = (event: KeyboardEvent) => {
			if (event.ctrlKey && event.shiftKey && event.code === 'KeyH') {
				event.preventDefault();
				goto('/history');
				return;
			}

			if (event.code === 'Space' && !isRecording && !isSpeaking) {
				event.preventDefault();
				startRecording();
			}
		};

		const handleKeyUp = (event: KeyboardEvent) => {
			if (event.code === 'Space' && isRecording) {
				event.preventDefault();
				stopRecording();
			}
		};

		window.addEventListener('keydown', handleKeyDown);
		window.addEventListener('keyup', handleKeyUp);

		return () => {
			clearInterval(clockInterval);
			window.removeEventListener('keydown', handleKeyDown);
			window.removeEventListener('keyup', handleKeyUp);
			if (animationInterval) clearInterval(animationInterval);
			if (currentAudio) {
				currentAudio.pause();
				currentAudio = null;
			}
			stopProcessingAnimation();
		};
	});

	function stopAllAnimation() {
		if (animationInterval) {
			clearInterval(animationInterval);
			animationInterval = null;
		}
		if (currentAudio) {
			currentAudio.pause();
			currentAudio = null;
		}
		stopProcessingAnimation();
		isSpeaking = false;
		isProcessing = false; // Reset processing flag
		spokenText = '';
		visibleChars = 0;
		showMurph = false;
		time = now();
	}

	async function animateTextWithAudio(text: string, audio: HTMLAudioElement) {
		// Clean text for display
		const cleanText = text.toUpperCase().replace(/[^A-Z0-9\s!?.,':]/g, '');
		const words = cleanText.split(' ').filter(w => w.length > 0);
		
		if (words.length === 0) {
			stopAllAnimation();
			return;
		}

		isSpeaking = true;
		showMurph = false;

		const totalChars = cleanText.replace(/\s/g, '').length;
		const audioDuration = audio.duration;
		
		// Calculate timing
		const charDelay = Math.max(25, Math.min(80, (audioDuration * 1000) / totalChars));
		
		console.log('Animation config:', {
			totalChars,
			audioDuration,
			charDelay,
			totalWords: words.length
		});

		let wordIndex = 0;
		
		while (wordIndex < words.length && isSpeaking) {
			// Get next chunk
			const chunk = words.slice(wordIndex, wordIndex + MAX_WORDS_PER_SCREEN);
			const chunkText = chunk.join(' ');
			
			console.log('Displaying chunk:', chunkText);
			
			// Reset for new chunk
			spokenText = chunkText;
			visibleChars = 0;

			// Animate character by character
			for (let i = 0; i <= chunkText.length; i++) {
				if (!isSpeaking) break;
				
				visibleChars = i;
				
				// Dynamic delay based on character type
				const currentChar = chunkText[i];
				let delay = charDelay;
				
				if (currentChar === ' ') {
					delay = charDelay / 4;
				} else if (['.', '!', '?'].includes(currentChar)) {
					delay = charDelay * 2;
				} else if ([',', ':'].includes(currentChar)) {
					delay = charDelay * 1.5;
				}
				
				await new Promise(r => setTimeout(r, delay));
			}

			wordIndex += MAX_WORDS_PER_SCREEN;
			
			// Brief pause between chunks
			if (wordIndex < words.length && isSpeaking) {
				await new Promise(r => setTimeout(r, 400));
			}
		}

		// Hold final text briefly
		if (isSpeaking) {
			await new Promise(r => setTimeout(r, 1500));
		}

		// Reset to clock
		stopAllAnimation();
	}

	async function startRecording() {
		try {
			showMurph = true;
			time = { hours: 'MU', minutes: 'RP', seconds: 'H!' };

			const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
			isRecording = true;
			audioChunks = [];

			mediaRecorder = new MediaRecorder(stream);

			mediaRecorder.ondataavailable = (event: BlobEvent) => {
				audioChunks.push(event.data);
			};

			mediaRecorder.onstop = async () => {
				const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
				const formData = new FormData();
				formData.append('audio', audioBlob, 'recording.webm');

				isProcessing = true; // Start processing state
				showMurph = true; // Keep MURPH display visible
				animateProcessing(); // Start the animation

				try {
					const response = await fetch('http://localhost:8000/voice-chat', {
						method: 'POST',
						body: formData
					});

					if (response.ok) {
						const responseBlob = await response.blob();
						
						// Get text from backend
						let responseText = 'MURPH IS HERE';
						try {
							const textResponse = await fetch('http://localhost:8000/last-response');
							if (textResponse.ok) {
								const data = await textResponse.json();
								responseText = data.text || responseText;
								console.log('Response text:', responseText);
							}
						} catch (e) {
							console.error('Could not fetch response text:', e);
						}

						const audioUrl = URL.createObjectURL(responseBlob);
						const audio = new Audio(audioUrl);
						currentAudio = audio;
						
						// Wait for audio to be ready
						audio.onloadedmetadata = () => {
							console.log('Audio duration:', audio.duration);
							
							// Stop processing animation before playing
							stopProcessingAnimation();
							
							// Start animation AFTER audio starts playing
							audio.play().then(() => {
								animateTextWithAudio(responseText, audio);
							}).catch(err => {
								console.error('Audio playback failed:', err);
								stopAllAnimation();
							});
						};

						audio.onerror = (e) => {
							console.error('Audio error:', e);
							stopAllAnimation();
							URL.revokeObjectURL(audioUrl);
						};

						audio.onended = () => {
							console.log('Audio ended');
							URL.revokeObjectURL(audioUrl);
							// Give a moment to finish text animation
							setTimeout(() => {
								if (isSpeaking) stopAllAnimation();
							}, 500);
						};
					} else {
						console.error('Server response not OK:', response.status);
						stopAllAnimation();
					}
				} catch (err) {
					console.error('Failed to get AI voice response:', err);
					stopAllAnimation();
				}

				// Stop microphone
				mediaRecorder?.stream.getTracks().forEach(track => track.stop());
			};

			mediaRecorder.start();
		} catch (err) {
			console.error('Error accessing microphone:', err);
			showMurph = false;
			isRecording = false;
		}
	}

	function stopRecording() {
		if (mediaRecorder && isRecording) {
			mediaRecorder.stop();
			isRecording = false;
		}
	}
</script>

<svelte:head>
	<title>Murph</title>
</svelte:head>

<div id="app" class:murph-active={showMurph} class:speaking={isSpeaking}>
	{#if isSpeaking}
		<TextDisplay text={spokenText} {visibleChars} />
	{:else if isProcessing}
		<Clock time={processingTime} />
	{:else}
		<Clock {time} />
	{/if}
</div>

<div class="instructions">
	{#if !isSpeaking && !isProcessing && !isRecording}
		<p>Press and hold <kbd>SPACE</kbd> to speak</p>
		<p class="shortcut">
			Press <kbd>Ctrl</kbd> + <kbd>Shift</kbd> + <kbd>H</kbd> for history
		</p>
	<!-- {:else if isRecording}
		<p class="recording-indicator">Listening...</p> -->
	{/if}
</div>

<style>
	#app {
		display: flex;
		justify-content: center;
		align-items: center;
		min-height: 100vh;
		padding: 2rem;
	}

	#app.speaking {
		align-items: center;
	}

	.instructions {
		position: fixed;
		bottom: 2rem;
		left: 50%;
		transform: translateX(-50%);
		text-align: center;
		color: oklch(60% 0 0);
	}

	.shortcut {
		font-size: 0.9em;
		color: oklch(50% 0 0);
		margin-top: 0.5rem;
	}

	.shortcut kbd {
		padding: 3px 8px;
		font-size: 0.85em;
	}

	/* .speaking-indicator {
		color: oklch(70% 0.15 150);
		animation: pulse 1.5s ease-in-out infinite;
	} */

	/* .recording-indicator {
		color: oklch(70% 0.15 30);
		animation: pulse 1s ease-in-out infinite;
	} */

	/* .processing-indicator {
		color: oklch(60% 0.15 240);
		animation: pulse 1.2s ease-in-out infinite;
	} */

	@keyframes pulse {
		0%, 100% { opacity: 0.6; }
		50% { opacity: 1; }
	}

	kbd {
		background-color: oklch(30% 0 0);
		border: 1px solid oklch(40% 0 0);
		border-radius: 3px;
		box-shadow: 0 1px 0 rgba(0, 0, 0, 0.2);
		color: var(--hand-color, oklch(94.75% 0.1916 119.15));
		transition: color 250ms ease-out;
		display: inline-block;
		font-family: monospace;
		font-size: 0.9em;
		padding: 4px 12px;
		white-space: nowrap;
	}
</style>