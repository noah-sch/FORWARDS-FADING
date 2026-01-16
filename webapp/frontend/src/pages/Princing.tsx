import { useState } from "react";

import Section from "../components/globals/Section";
import TitleSubSection from "../components/globals/TitleSubSection";
import PricingPlan from "../components/pricing/PricingPlan";
import ToggleBtn from "../components/globals/ToggleBtn";

import { PRICINGPLANSTART, PRICINGPLANPRO, PRICINGPLANONPREMISE } from "../content/PRICINGPLANS";

export default function Pricing() {

    const [modePricing, setModePricing] = useState<"monthly" | "yearly">("monthly");

    return (
        <>
            <div className="w-full flex flex-col">
                <Section>
                    <TitleSubSection title={["Simple and", "transparent pricing"]} underTitle={["Find the plan that fits you the most"]} />

                    <div className="w-full flex flex-row items-center justify-center gap-2">
                        <div className="text-xl text-c1">
                            Monthly
                        </div>
                        <ToggleBtn cat1={"monthly"} cat2={"yearly"} toToggle={modePricing} setToToggle={setModePricing} />
                        <div className="text-xl text-c1">
                            Yearly
                        </div>
                        <div className="rounded-full text-sm text-c1 bg-c3 px-2 py-1">
                            20% OFF
                        </div>
                    </div>

                    <div className="w-full grid grid-cols-12 gap-4">
                        {[PRICINGPLANSTART, PRICINGPLANPRO, PRICINGPLANONPREMISE].map((plan, index) => (
                            <>
                                <div key={index}
                                    className="col-span-4"
                                >
                                    <PricingPlan pricingPlan={plan} mode={modePricing}/>
                                </div>
                            </>
                        ))}
                    </div>
                </Section>
                <Section>
                    <div>
                        
                    </div>
                </Section>
            </div>
        </>
    )
}