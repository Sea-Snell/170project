var items = document.getElementsByClassName("table")[0].rows;
var dict = {};
for (var i = 0; i < items.length; i++) {
	dict[items[i].cells[0].textContent] = parseInt(items[i].cells[1].innerHTML);
}
function downloadObjectAsJson(exportObj, exportName){
    var dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(exportObj));
    var downloadAnchorNode = document.createElement('a');
    downloadAnchorNode.setAttribute("href",     dataStr);
    downloadAnchorNode.setAttribute("download", exportName + ".json");
    document.body.appendChild(downloadAnchorNode); // required for firefox
    downloadAnchorNode.click();
    downloadAnchorNode.remove();
  }

downloadObjectAsJson(dict, 'leaders')