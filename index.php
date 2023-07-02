<?php
if(isset($_GET["deploy"])){
    if($_GET["deploy"]==1){
        echo shell_exec("python3.6 wsgi.py");
        echo "<a href='http://ocr.varbin.com:8080/upload/ocr' target='_blank'>Click here to check</a>";
    }
}
else{
    ?>

        <?php
}


?><br/>
Click here to deploy.
<form>
    <button class="" name="deploy" value="1">Deploy Code</button> 
</form> 