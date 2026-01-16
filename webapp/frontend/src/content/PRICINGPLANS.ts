import type { pricingPlan } from "../types/pricingPlan";

export const PRICINGPLANSTART: pricingPlan = {
    title: "Free",
    monthly: 0,
    yearly: 0,
    description: "Perfect for non-frequently personnal use, or AI exploration.",
    btn: "Use for Free",
    btnType: 2,
    features: [
        "Up to 50 daily prompts or a maximum of 100k daily tokens",
        "Basic AI responses",
        "Context and memory (limited)",
        "Standard support",
        "Secure conversation storage"
    ],
    tag: null
}

export const PRICINGPLANPRO: pricingPlan = {
    title: "Pro",
    monthly: 15,
    yearly: 12,
    description: "Ideal for highly personnal use needing speed and large context.",
    btn: "Upgrade to Pro",
    btnType: 1,
    features: [
        "Up to 200 daily prompts or a maximum of 500k daily tokens",
        "Advanced responses",
        "Latest models availability",
        "Deeper context and memory",
        "Standard support",
        "Secure conversation storage"
    ],
    tag: "Popular",
}

export const PRICINGPLANONPREMISE: pricingPlan = {
    title: "Local ",
    monthly: "Custom",
    yearly: "Custom",
    description: "Thought for on-premise/private cloud setup for personal or enterprise premium experience.",
    btn: "Contact sales",
    btnType: 2,
    features: [
        "Unlimited requests",
        "Offline usage",
        "Custom integrations",
        "Enterprise plans",
        "Rag enhancing",
        "Complete pipelining",
        "24/7 support"
    ],
    tag: null,
}