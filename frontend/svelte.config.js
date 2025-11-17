// svelte.config.js

import adapter from '@sveltejs/adapter-auto';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

/** @type {import('@sveltejs/kit').Config} */
const config = {
    preprocess: vitePreprocess(),

    kit: {
        adapter: adapter(),
        
		alias: {
			$lib: './src/lib'
		}

    },

    // --- ADD THIS VITE CONFIGURATION BLOCK ---
    vite: {
        server: {
            fs: {
                // Allow serving files from the picovoice package directory
                allow: ['node_modules/@picovoice']
            }
        }
    }
    // ------------------------------------------
};

export default config;