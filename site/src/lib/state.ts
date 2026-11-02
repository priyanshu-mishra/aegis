// One control propagates: the promised level α is shared by every linked exhibit on the page.
import { writable } from "svelte/store";

export const alpha = writable(0.05);

/** Promise levels offered by the sliders, as whole percentages (α = (100 − level) / 100). */
export const LEVELS = { min: 70, max: 99 };
