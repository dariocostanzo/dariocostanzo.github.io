// Check Off Specific Todos By Clicking
// $("li").click(function(){
	$("ul").on("click", "li", function(){ //listening to the li inside the ul
		$(this).toggleClass("completed");
	});
 
 //  //if li is gray
 //  if($(this).css("color") === "rgb(128, 128, 128)"){
 //  	//turn it black
 //  	$(this).css({
	// 	color: "black",
	// 	textDecoration: "none"
	// });		
 //  }

 //  //else        
 //    else{
 //    	//turn it gray
	// $(this).css({
	// 	color: "gray",
	// 	textDecoration: "line-through"
	// });
 //  }

 	//$(this).toggleClass("completed");
//});

//Click on X to delete Todo
//$("span").click(function(event){
$("ul").on("click", "span", function(event){
	$(this).parent().fadeOut(500, function(){//this = span, .parent = li, .remove = remove the entire li
		$(this).remove(); //refers to the li
	}); 
	event.stopPropagation(); //stop from bubbling up to other elements
});

//add a listener to inout
$("input[type='text']").keypress(function(event){
	if(event.which === 13){ //13 refers to enter
		var todoText = $(this).val(); //grabbing new todo text from input
		$(this).val(""); //empty input when I type a new todo
		//create a new li and add to ul
		$("ul").append("<li><span><i class='fa fa-trash'></i></span> " + todoText + "</li>");
	}
});

// 1. Append method
// $("ul").append("<li><span>X</span> " + todoText + "</li>");
// Which can take a string of HTML and append those elements to whatever we select

// 2. Using "On" rather than ".click"
// $("ul").on("click", "span", function(event){
// $("ul").on("click", "li", function(){
// and the second argument that specifies li's that may or may not append
// on the page when onloaded 