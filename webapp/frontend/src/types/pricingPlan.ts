export type pricingPlan = {
    title: string;
    monthly: number | "Custom";
    yearly: number | "Custom";
    description: string;
    btn: string;
    btnType: 1 | 2;
    features: string[];
    tag: string | null;
}