import Section from "../components/globals/Section";
import Title from "../components/globals/titles/Title";
import TitleSubSection from "../components/globals/TitleSubSection";

export default function Pricing() {
    return (
        <>
            <div className="w-full flex flex-col">
                <Section>
                    <TitleSubSection title={["Simple and", "transparent pricing"]} underTitle={["Find the plan that fits you the most"]} />
                </Section>
                <div>

                </div>
            </div>
        </>
    )
}