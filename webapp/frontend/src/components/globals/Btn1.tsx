type Btn1Props = {
    text: string;
}

export default function Btn1( {text}: Btn1Props ){
    return (
        <>
            <div className="text-sm font-medium text-c4 rounded-md bg-c1 px-2 py-1.5 cursor-pointer hover:bg-c1hov">
                {text}
            </div>
        </>
    )
}