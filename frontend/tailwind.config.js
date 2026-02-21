/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            colors: {
                bb: {
                    bg: '#040605',
                    panel: '#131919',
                    border: '#242F35',
                    accent: '#1B353F',
                    gold: '#948161',
                    text: '#989A8F'
                }
            },
            fontFamily: {
                serif: ['"Inria Serif"', 'serif'],
            }
        },
    },
    plugins: [],
}
