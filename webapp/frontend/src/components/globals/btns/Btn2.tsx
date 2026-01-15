type Btn2Props = {
    text: string;
}

export default function Btn2( {text}: Btn2Props ){
    return (
        <>
            <div className="text-sm font-medium text-c1 rounded-md bg-c3 px-2 py-1.5 cursor-pointer hover:bg-c3hov">
                {text}
            </div>
        </>
    )
}