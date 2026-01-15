import Title from "./titles/Title";
import UnderTitle from "./titles/UnderTitle";

type TitleSubSectionProps = {
    title: string[];
    underTitle: string[];
}

export default function TitleSubSection ( {title, underTitle}: TitleSubSectionProps ) {
    return (
        <>
            <div className="w-full flex flex-col items-center justify-start gap-2">
                <Title title={title} />
                <UnderTitle underTitle={underTitle} />
            </div>
        </>
    )
}