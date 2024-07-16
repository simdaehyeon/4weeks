document.addEventListener("DOMContentLoaded", function() {
    // 이미지 업로드 시 미리보기 표시
    document.getElementById('image-upload').addEventListener('change', function(event) {
        const file = event.target.files[0];
        const reader = new FileReader();
        //이미지 로드시 공간 확보
        reader.onload = function(e) {
            const img = document.getElementById('image-preview');
            img.src = e.target.result;
            img.style.display = 'block';
        };

        if (file) {
            reader.readAsDataURL(file);
        }
    });

    const uploadForm = document.getElementById("upload-form");
    //분석 시작 버튼 클릭 시  이벤트 발생
    uploadForm.addEventListener("submit", function(event) {
        event.preventDefault(); 

        const fileInput = document.getElementById('image-upload');
        const file = fileInput.files[0];

        const stepBox1 = document.querySelector(".step-box-1");
        stepBox1.classList.add("slide-out-left");


        const stepNumber2 = document.querySelector(".step-number2");
        stepNumber2.style.display = "none";


        const stepBox2 = document.querySelector(".step-box-2");
        stepBox2.style.backgroundColor ="white";



        stepBox1.addEventListener("animationend", function() {
            stepBox1.style.display = "none";



        }, { once: true });

    });
});
