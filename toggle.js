
// var button = document.querySelector("button");
// // var isPurple = false;

// // // button.addEventListener("click", function(){
// // // 	if(isPurple){
// // // 		document.body.style.background = "white";
// // // 	} else {
// // // 		document.body.style.background = "purple";
// // // 	}
// // // 	isPurple = !isPurple;

// button.addEventListener("click",function(){
// 	document.body.classList.toggle("purple");
// });

var button = document.querySelector("#picture");
// var isPurple = false;


// button.addEventListener("click", function(){
// 	if(isPurple){
// 		document.body.style.background = "white";
// 	} else {
// 		document.body.style.background = "purple";
// 	}
// 	isPurple = !isPurple;
// });


button.addEventListener("click", function(){
	document.body.classList.toggle("purple");
});

var picture = document.getElementById("picture");

picture.addEventListener("mouseover", function(){
	picture.style.color = "yellow";
});
picture.addEventListener("mouseout", function(){
	picture.style.color = "white";
});