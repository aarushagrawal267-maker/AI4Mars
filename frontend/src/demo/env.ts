/** True in the browser edition (static build that runs the models in the browser, no Python server). */
export const IS_DEMO: boolean = import.meta.env.VITE_DEMO === '1';
