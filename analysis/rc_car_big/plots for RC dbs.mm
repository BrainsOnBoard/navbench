<map version="freeplane 1.9.13">
<!--To view this file, download free mind mapping software Freeplane from https://www.freeplane.org -->
<node TEXT="plots for RC dbs" FOLDED="false" ID="ID_696401721" CREATED="1610381621824" MODIFIED="1646043346419" STYLE="oval">
<font SIZE="18"/>
<hook NAME="AutomaticEdgeColor" COUNTER="9" RULE="ON_BRANCH_CREATION"/>
<hook NAME="MapStyle">
    <properties edgeColorConfiguration="#808080ff,#ff0000ff,#0000ffff,#00ff00ff,#ff00ffff,#00ffffff,#7c0000ff,#00007cff,#007c00ff,#7c007cff,#007c7cff,#7c7c00ff" associatedTemplateLocation="template:/standard-1.6.mm" followedTemplateLocation="template:/standard-1.6.mm" followedMapLastTime="1647966556000" fit_to_viewport="false"/>

<map_styles>
<stylenode LOCALIZED_TEXT="styles.root_node" STYLE="oval" UNIFORM_SHAPE="true" VGAP_QUANTITY="24 pt">
<font SIZE="24"/>
<stylenode LOCALIZED_TEXT="styles.predefined" POSITION="right" STYLE="bubble">
<stylenode LOCALIZED_TEXT="default" ID="ID_271890427" ICON_SIZE="12 pt" COLOR="#000000" STYLE="fork">
<arrowlink SHAPE="CUBIC_CURVE" COLOR="#000000" WIDTH="2" TRANSPARENCY="200" DASH="" FONT_SIZE="9" FONT_FAMILY="SansSerif" DESTINATION="ID_271890427" STARTARROW="NONE" ENDARROW="DEFAULT"/>
<font NAME="SansSerif" SIZE="10" BOLD="false" ITALIC="false"/>
<richcontent CONTENT-TYPE="plain/auto" TYPE="DETAILS"/>
<richcontent TYPE="NOTE" CONTENT-TYPE="plain/auto"/>
</stylenode>
<stylenode LOCALIZED_TEXT="defaultstyle.details"/>
<stylenode LOCALIZED_TEXT="defaultstyle.attributes">
<font SIZE="9"/>
</stylenode>
<stylenode LOCALIZED_TEXT="defaultstyle.note" COLOR="#000000" BACKGROUND_COLOR="#ffffff" TEXT_ALIGN="LEFT"/>
<stylenode LOCALIZED_TEXT="defaultstyle.floating">
<edge STYLE="hide_edge"/>
<cloud COLOR="#f0f0f0" SHAPE="ROUND_RECT"/>
</stylenode>
<stylenode LOCALIZED_TEXT="defaultstyle.selection" BACKGROUND_COLOR="#4e85f8" BORDER_COLOR_LIKE_EDGE="false" BORDER_COLOR="#4e85f8"/>
</stylenode>
<stylenode LOCALIZED_TEXT="styles.user-defined" POSITION="right" STYLE="bubble">
<stylenode LOCALIZED_TEXT="styles.topic" COLOR="#18898b" STYLE="fork">
<font NAME="Liberation Sans" SIZE="10" BOLD="true"/>
</stylenode>
<stylenode LOCALIZED_TEXT="styles.subtopic" COLOR="#cc3300" STYLE="fork">
<font NAME="Liberation Sans" SIZE="10" BOLD="true"/>
</stylenode>
<stylenode LOCALIZED_TEXT="styles.subsubtopic" COLOR="#669900">
<font NAME="Liberation Sans" SIZE="10" BOLD="true"/>
</stylenode>
<stylenode LOCALIZED_TEXT="styles.important" ID="ID_67550811">
<icon BUILTIN="yes"/>
<arrowlink COLOR="#003399" TRANSPARENCY="255" DESTINATION="ID_67550811"/>
</stylenode>
</stylenode>
<stylenode LOCALIZED_TEXT="styles.AutomaticLayout" POSITION="right" STYLE="bubble">
<stylenode LOCALIZED_TEXT="AutomaticLayout.level.root" COLOR="#000000" STYLE="oval" SHAPE_HORIZONTAL_MARGIN="10 pt" SHAPE_VERTICAL_MARGIN="10 pt">
<font SIZE="18"/>
</stylenode>
<stylenode LOCALIZED_TEXT="AutomaticLayout.level,1" COLOR="#0033ff">
<font SIZE="16"/>
</stylenode>
<stylenode LOCALIZED_TEXT="AutomaticLayout.level,2" COLOR="#00b439">
<font SIZE="14"/>
</stylenode>
<stylenode LOCALIZED_TEXT="AutomaticLayout.level,3" COLOR="#990000">
<font SIZE="12"/>
</stylenode>
<stylenode LOCALIZED_TEXT="AutomaticLayout.level,4" COLOR="#111111">
<font SIZE="10"/>
</stylenode>
<stylenode LOCALIZED_TEXT="AutomaticLayout.level,5"/>
<stylenode LOCALIZED_TEXT="AutomaticLayout.level,6"/>
<stylenode LOCALIZED_TEXT="AutomaticLayout.level,7"/>
<stylenode LOCALIZED_TEXT="AutomaticLayout.level,8"/>
<stylenode LOCALIZED_TEXT="AutomaticLayout.level,9"/>
<stylenode LOCALIZED_TEXT="AutomaticLayout.level,10"/>
<stylenode LOCALIZED_TEXT="AutomaticLayout.level,11"/>
</stylenode>
</stylenode>
</map_styles>
</hook>
<node TEXT="legend" POSITION="right" ID="ID_1728326240" CREATED="1646043554465" MODIFIED="1646043555928">
<edge COLOR="#00ff00"/>
</node>
<node TEXT="are arrows correct for training route?" POSITION="right" ID="ID_1041967436" CREATED="1646045013972" MODIFIED="1646045024806">
<edge COLOR="#007c00"/>
<node TEXT="maybe use GPS for headings cf. IMU?" ID="ID_839470124" CREATED="1646045411022" MODIFIED="1646045435581">
<node TEXT="(plot GPS headings vs IMU?)" ID="ID_1970278285" CREATED="1646045462614" MODIFIED="1646045469303"/>
</node>
</node>
<node TEXT="boxplots of error (systematic deviations?)" POSITION="right" ID="ID_739213545" CREATED="1646043346429" MODIFIED="1646043367271">
<edge COLOR="#ff0000"/>
</node>
<node TEXT="arrange by distance" POSITION="right" ID="ID_1597079316" CREATED="1646043358174" MODIFIED="1646043409904">
<edge COLOR="#0000ff"/>
<node TEXT="scatter of error vs distance" ID="ID_527148418" CREATED="1646043414286" MODIFIED="1646043420009">
<node TEXT="different colours" ID="ID_1518149154" CREATED="1646043540519" MODIFIED="1646043545224"/>
</node>
<node TEXT="&quot;duff&quot; routes? e.g. bad heading_offset" ID="ID_650247892" CREATED="1646043520660" MODIFIED="1646043535712"/>
</node>
<node TEXT="error vs distance along route" POSITION="right" ID="ID_839822709" CREATED="1646043581935" MODIFIED="1646043674042">
<edge COLOR="#ff00ff"/>
<node TEXT="(distance along training route?)" ID="ID_167992874" CREATED="1646043634568" MODIFIED="1646043643785"/>
</node>
<node TEXT="ask Thomas re Google Maps" POSITION="right" ID="ID_1406112499" CREATED="1646043719817" MODIFIED="1646043729441">
<edge COLOR="#00ffff"/>
</node>
<node TEXT="figure out sources of errors" POSITION="right" ID="ID_1427953810" CREATED="1646043804897" MODIFIED="1646043809638">
<edge COLOR="#7c0000"/>
<node TEXT="&quot;bad&quot; training images?" ID="ID_1187054354" CREATED="1646043834363" MODIFIED="1646043839452">
<node TEXT="pointing at ground?" ID="ID_1314481874" CREATED="1646043929521" MODIFIED="1646043934058"/>
</node>
<node TEXT="&quot;bad&quot; areas on training route" ID="ID_864152567" CREATED="1646043913041" MODIFIED="1646043919922">
<node TEXT="difficult?" ID="ID_740721826" CREATED="1646043919925" MODIFIED="1646043921826"/>
</node>
<node TEXT="spurious errors" ID="ID_1040502741" CREATED="1646043839918" MODIFIED="1646043956922">
<node TEXT="right heading, but get error anyway" ID="ID_1879930341" CREATED="1646043956935" MODIFIED="1646043958155"/>
</node>
<node TEXT="is test image matching nearest train image?" ID="ID_1016016434" CREATED="1646044025034" MODIFIED="1646044042178">
<node TEXT="and what about at best-matching heading?" ID="ID_1163421942" CREATED="1646044110916" MODIFIED="1646044124388"/>
</node>
<node TEXT="IDFs along route" ID="ID_25875970" CREATED="1646044181491" MODIFIED="1646044187090"/>
<node TEXT="try figuring out how to categorise points as good and bad" ID="ID_471997943" CREATED="1646044834050" MODIFIED="1646044858015">
<node TEXT="places where it goes off course are the most interesting" ID="ID_1978230552" CREATED="1646044875997" MODIFIED="1646044888025"/>
</node>
</node>
<node TEXT="preprocessing" POSITION="right" ID="ID_269079376" CREATED="1646044008609" MODIFIED="1646044012843">
<edge COLOR="#00007c"/>
<node TEXT="filter out sky" ID="ID_1365731425" CREATED="1646044012846" MODIFIED="1646044015067"/>
<node TEXT="etc." ID="ID_1101086474" CREATED="1646045163184" MODIFIED="1646045163955"/>
</node>
<node TEXT="new todo list" POSITION="right" ID="ID_1537672976" CREATED="1648056306543" MODIFIED="1648056312207">
<edge COLOR="#7c007c"/>
<node TEXT="pick positions where GPS is definitely good and check that the headings are good" ID="ID_1076913655" CREATED="1648056312213" MODIFIED="1648056371037">
<node TEXT="e.g. GPS quality" ID="ID_1104818359" CREATED="1648056371046" MODIFIED="1648056376431"/>
</node>
<node TEXT="bad positions:" ID="ID_1026000879" CREATED="1648056477038" MODIFIED="1648056479983">
<node TEXT="plot: view + snap + diff + ridf and then also do for &quot;nearest&quot; snapshot" ID="ID_479934798" CREATED="1648056479988" MODIFIED="1648056541763"/>
</node>
</node>
</node>
</map>
