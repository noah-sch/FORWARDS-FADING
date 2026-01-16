import type { pricingPlan } from "../../types/pricingPlan";
import Btn1 from "../globals/btns/Btn1";
import Btn2 from "../globals/btns/Btn2";

type PricingPlanProps = {
    pricingPlan: pricingPlan;
    mode: "monthly" | "yearly";
}

export default function PricingPlan( {pricingPlan, mode}: PricingPlanProps ) {
    return (
        <>
            <div className="w-full h-full flex flex-col items-start justify-start rounded-lg p-6 gap-4 bg-c5">

                {/* Title */}
                <div className="w-full flex flex-row items-center justify-between">
                    <div className="text-lg text-c1">
                        {pricingPlan.title}
                    </div>
                    {pricingPlan.tag && (
                        <>
                            <div className="rounded-full text-sm text-c1 bg-c3 px-2 py-1">
                                {pricingPlan.tag}
                            </div>
                        </>
                    )}
                </div>

                {/* Price */}
                <div className="w-full flex flex-row items-end justify-start">
                    {pricingPlan.monthly === "Custom" ? (
                        <>
                            <div className="font-serif font-semibold text-4xl text-c1">
                                {pricingPlan.monthly}
                            </div>
                        </>
                    ) : (
                        <>
                            <div className="font-serif font-semibold text-4xl text-c1">
                                €{mode === "monthly" ? pricingPlan.monthly : pricingPlan.yearly}
                            </div>
                            <div className="text-sm text-c2">
                                /month
                            </div>
                        </>
                    )}
                </div>

                {/* Description */}
                <div className="w-full text-c2 text-sm">
                    {pricingPlan.description}
                </div>

                {/* Btn */}
                {pricingPlan.btnType === 1 ? (
                    <>
                        <Btn1 text={pricingPlan.btn} />
                    </>
                ) : (
                    <>
                        <Btn2 text={pricingPlan.btn} />
                    </>
                )}
                
                {/* Feature bar */}
                <div className="w-full flex flex-row items-center justify-center gap-2">
                    <div className="w-full border border-c2/10"/>
                    <div className="text-sm text-c2">
                        Features
                    </div>
                    <div className="w-full border border-c2/10"/>
                </div>

                {/* Features */}
                <div className="w-full flex flex-col items-start justify-center gap-2">
                    {pricingPlan.features.map((feature, index) => (
                        <>
                            <div key={index}
                                className="flex flex-row text-sm text-c2 gap-2"
                            >
                                ✓
                                <div>
                                    {feature}
                                </div>
                            </div>
                        </>
                    ))}
                </div>
            </div>
        </>
    )
}