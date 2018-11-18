// var firstLI = document.querySelector("li"); //select the first li
var lis = document.querySelectorAll("li"); //selec all li

// we need a for loop to apply the changes to whole the lis

for (var i = 0; i<lis.length; i++){//to add an event listener to each li
	lis[i].addEventListener("mouseover", function(){//add event listenere to li
  		// this.style.color = "green"; //this refer to the item that the event was triggered on
  		this.classList.add("selected");
  	});
    lis[i].addEventListener("mouseout", function(){
    	// this.style.color = "black";
    	this.classList.remove("selected");
	}); // li will get black once the mouse is out.

	//adding a click listener
	lis[i].addEventListener("click",function(){
		this.classList.toggle("done");		
	});
}


// firstLI.addEventListener("mouseover", function(){//add event listenere to li
//   firstLI.style.color = "green";
// }); //will make the li green, but without changing it back to black so we need a mouseout

// firstLI.addEventListener("mouseout", function(){
//    firstLI.style.color = "black";
// }); // li will get black once the mouse is out.

